# windowing_wideband.py

import csv
import os
import sys

import numpy as np

import config
from load_stieger import load_stieger_subject


# =========================================================
# 1) BASIT AYARLAR
# =========================================================

TARGET_SFREQ = config.TARGET_SFREQ
BROAD_LOW_FREQ = config.BROAD_LOW_FREQ
BROAD_HIGH_FREQ = config.BROAD_HIGH_FREQ
WINDOW_SIZE_SEC = config.WINDOW_SIZE_SEC
STRIDE_SEC = config.STRIDE_SEC

# Wideband pencereleme, 5 canonical band CSP deneyinin ortak ham girdisini uretir.
# Bantlara ayirma burada degil, her LOSO fold'u icinde train/test ayrimindan sonra yapilir.
MOTOR_CHANNELS = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4"
]


# =========================================================
# 2) LABEL TABLOSUNU OKU
# =========================================================

def get_session_from_cli():
    """
    Terminalden session numarasini alir.

    Ornek:
    python windowing_wideband.py 1
    """
    if len(sys.argv) < 2:
        raise ValueError("Kullanim: python windowing_wideband.py <session_id>")

    session_name = str(sys.argv[1])
    return session_name


def read_label_table(session_name):
    """
    build_labels.py ciktisini okur.
    """
    file_path = os.path.join(
        config.OUTPUT_DIR,
        "label_tables",
        f"session_{session_name}_labels.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label tablosu bulunamadi: {file_path}")

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        raise ValueError("Label tablosu bos gorunuyor.")

    return rows


# =========================================================
# 3) SADE SESSION RAW CEKME
# =========================================================

def get_session_raw(subject_data, session_name):
    """
    Istenen session icindeki ilk run'in raw nesnesini dondurur.
    """
    session_name = str(session_name)

    if session_name not in subject_data:
        raise ValueError(f"Session {session_name} bulunamadi.")

    runs = subject_data[session_name]
    run_names = list(runs.keys())

    if len(run_names) == 0:
        raise ValueError(f"Session {session_name} icinde run yok.")

    run_name = run_names[0]
    raw = runs[run_name]

    return raw, run_name


# =========================================================
# 4) PREPROCESSING (WIDEBAND)
# =========================================================

def preprocess_raw_wideband(raw):
    """
    Canonical 5-band CSP deneyi icin wideband pencere cikarimi:
    - motor kanallari sec
    - average reference uygula
    - 250 Hz'e resample et
    - genis bant filtre uygula (0.5-50)
    """
    raw_proc = raw.copy()

    # Motor kanal secimi, dar bant baseline ile ayni elektrot uzayini korur.
    # Boylece bandpower, CSP ve 5-band CSP deneyleri ayni bolge uzerinden karsilastirilir.
    existing_channels = [ch for ch in MOTOR_CHANNELS if ch in raw_proc.ch_names]
    if len(existing_channels) == 0:
        raise ValueError("Motor kanal listesinde raw icinde bulunan kanal yok.")

    # Genis bant filtre, daha sonra delta/theta/alpha/beta/gamma bantlarina ayrilacak sinyali hazirlar.
    # Bu dosya supervised bir islem fit etmez; CSP fit'i egitim fold'u icinde kalmalidir.
    raw_proc.pick(existing_channels)
    raw_proc.set_eeg_reference("average", projection=False)
    raw_proc.resample(TARGET_SFREQ)
    raw_proc.filter(BROAD_LOW_FREQ, BROAD_HIGH_FREQ, verbose=False)

    return raw_proc


# =========================================================
# 5) PENCERE SAYISI / SINIR KONTROLU
# =========================================================

def segment_too_short(start_sec, end_sec):
    """
    Segment pencere cikarmaya yetiyor mu kontrol eder.
    """
    return (end_sec - start_sec) < WINDOW_SIZE_SEC


# =========================================================
# 6) GERCEK EEG PENCERELERINI CIKAR
# =========================================================

def extract_windows_from_segment(raw_proc, segment_row, start_window_id):
    """
    Tek bir label segmentinden gercek EEG pencereleri cikarir.
    """
    windows_data = []
    windows_meta = []

    segment_id = int(segment_row["segment_id"])
    trial_id = int(segment_row["trial_id"])
    session_id = str(segment_row["session_id"])
    run_id = str(segment_row["run_id"])
    source_segment = str(segment_row["source_segment"])
    label = int(segment_row["label"])
    cue_label = str(segment_row["cue_label"])

    start_sec = float(segment_row["start_sec"])
    end_sec = float(segment_row["end_sec"])
    duration_sec = float(segment_row["duration_sec"])

    # Kisa segmentleri sessizce yok saymak yerine dropped CSV'ye nedeniyle yaziyoruz.
    # Bu, ITI/feedback dengesini ve veri kaybini sonradan denetlemeyi saglar.
    if segment_too_short(start_sec, end_sec):
        dropped_row = {
            "segment_id": segment_id,
            "trial_id": trial_id,
            "session_id": session_id,
            "run_id": run_id,
            "source_segment": source_segment,
            "label": label,
            "cue_label": cue_label,
            "segment_start_sec": start_sec,
            "segment_end_sec": end_sec,
            "segment_duration_sec": duration_sec,
            "reason": "segment_too_short_for_window"
        }
        return windows_data, windows_meta, dropped_row, start_window_id

    sfreq = raw_proc.info["sfreq"]
    window_id = start_window_id
    current_start = start_sec

    while current_start + WINDOW_SIZE_SEC <= end_sec + 1e-9:
        current_end = current_start + WINDOW_SIZE_SEC

        start_sample = int(round(current_start * sfreq))
        end_sample = int(round(current_end * sfreq))

        window_data = raw_proc.get_data(start=start_sample, stop=end_sample)
        expected_n_samples = int(round(WINDOW_SIZE_SEC * sfreq))

        if window_data.shape[1] != expected_n_samples:
            current_start += STRIDE_SEC
            continue

        windows_data.append(window_data)

        # Label, cue komutundan degil segment tipinden gelir: ITI=0, feedback=1.
        # cue_label sadece yorumlama icin metadata olarak pencerede tasinir.
        meta_row = {
            "window_id": window_id,
            "segment_id": segment_id,
            "trial_id": trial_id,
            "session_id": session_id,
            "run_id": run_id,
            "source_segment": source_segment,
            "label": label,
            "cue_label": cue_label,
            "window_start_sec": round(current_start, 4),
            "window_end_sec": round(current_end, 4),
            "window_duration_sec": WINDOW_SIZE_SEC,
            "window_start_sample": start_sample,
            "window_end_sample": end_sample
        }

        windows_meta.append(meta_row)
        window_id += 1
        current_start += STRIDE_SEC

    return windows_data, windows_meta, None, window_id


# =========================================================
# 7) TUM SEGMENTLER ICIN
# =========================================================

def build_all_windows(raw_proc, label_rows):
    """
    Tum label segmentlerinden gercek EEG pencereleri cikarir.
    """
    all_window_data = []
    all_window_meta = []
    dropped_rows = []

    next_window_id = 1

    for row in label_rows:
        window_data_list, window_meta_list, dropped_row, next_window_id = extract_windows_from_segment(
            raw_proc,
            row,
            next_window_id
        )

        all_window_data.extend(window_data_list)
        all_window_meta.extend(window_meta_list)

        if dropped_row is not None:
            dropped_rows.append(dropped_row)

    if len(all_window_data) == 0:
        raise ValueError("Hic pencere uretilemedi.")

    # X/y/metadata ayni ekleme sirasindan gelir.
    # Bu siralama korunmazsa model tahminleri yanlis pencere bilgisiyle eslesebilir.
    X = np.stack(all_window_data, axis=0)
    y = np.array([row["label"] for row in all_window_meta], dtype=int)

    return X, y, all_window_meta, dropped_rows


# =========================================================
# 8) KISA OZET
# =========================================================

def print_window_preview(window_meta, max_rows=10):
    """
    Ilk birkac pencereyi ekrana basar.
    """
    print("\n===== ILK WIDEBAND PENCERELER =====")

    limit = min(max_rows, len(window_meta))
    for i in range(limit):
        row = window_meta[i]
        print(
            f"window_id={row['window_id']} | "
            f"trial_id={row['trial_id']} | "
            f"type={row['source_segment']} | "
            f"label={row['label']} | "
            f"start={row['window_start_sec']} | "
            f"end={row['window_end_sec']}"
        )


def print_window_summary(X, y, dropped_rows):
    """
    Genel pencere ozetini ekrana basar.
    """
    n_label_0 = int(np.sum(y == 0))
    n_label_1 = int(np.sum(y == 1))

    print("\n===== WIDEBAND WINDOW OZETI =====")
    print("X shape:", X.shape)
    print("Toplam pencere sayisi:", len(y))
    print("Label 0 pencere sayisi:", n_label_0)
    print("Label 1 pencere sayisi:", n_label_1)
    print("Dislanan kisa segment sayisi:", len(dropped_rows))


# =========================================================
# 9) KAYIT
# =========================================================

def save_metadata_csv(window_meta, session_name):
    """
    Pencere metadata'sini CSV olarak kaydeder.
    """
    # Wideband metadata, normal windowing metadata'siyle karsilastirilabilir kalmalidir.
    # Bu dosya X/y satirlarinin hangi trial ve segmentten geldigini belgeler.
    save_dir = os.path.join(config.OUTPUT_DIR, "window_data_wideband")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_wideband_window_metadata.csv")

    fieldnames = list(window_meta[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window_meta)

    print(f"Metadata kaydedildi: {save_path}")


def save_dropped_segments_csv(dropped_rows, session_name):
    """
    Kisa kaldigi icin dislanan segmentleri kaydeder.
    """
    # Bos dropped listesi bile header ile kaydedilir.
    # Her session icin eleme kaydi olmasi raporlamada izlenebilirlik saglar.
    save_dir = os.path.join(config.OUTPUT_DIR, "window_data_wideband")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_wideband_dropped_segments.csv")

    if len(dropped_rows) == 0:
        fieldnames = [
            "segment_id", "trial_id", "session_id", "run_id", "source_segment",
            "label", "cue_label", "segment_start_sec", "segment_end_sec",
            "segment_duration_sec", "reason"
        ]
    else:
        fieldnames = list(dropped_rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dropped_rows)

    print(f"Dropped segmentler kaydedildi: {save_path}")


def save_npz(X, y, session_name):
    """
    Gercek EEG pencerelerini npz dosyasi olarak kaydeder.
    """
    # Bu NPZ, 5-band CSP ve feature selection deneylerinin ana girdisidir.
    # Kayit adi sabit tutuldugu icin batch ve ablation scriptleri ayni artefakti bulabilir.
    save_dir = os.path.join(config.OUTPUT_DIR, "window_data_wideband")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_wideband_windows.npz")
    np.savez_compressed(save_path, X=X, y=y)

    print(f"EEG wideband pencere verisi kaydedildi: {save_path}")


# =========================================================
# 10) ANA AKIS
# =========================================================

def main():
    subject_id = config.SUBJECT_ID
    session_name = get_session_from_cli()

    print(f"Subject {subject_id} yukleniyor...")
    subject_data = load_stieger_subject(subject_id)

    print(f"Session {session_name} seciliyor...")
    raw, run_name = get_session_raw(subject_data, session_name)

    del subject_data

    print("Label tablosu okunuyor...")
    label_rows = read_label_table(session_name)

    print("Wideband preprocessing uygulaniyor...")
    raw_proc = preprocess_raw_wideband(raw)

    print("Gercek EEG wideband pencereleri cikariliyor...")
    X, y, window_meta, dropped_rows = build_all_windows(raw_proc, label_rows)

    print_window_preview(window_meta)
    print_window_summary(X, y, dropped_rows)

    save_npz(X, y, session_name)
    save_metadata_csv(window_meta, session_name)
    save_dropped_segments_csv(dropped_rows, session_name)

    print("\nWideband windowing tamamlandi.")


if __name__ == "__main__":
    main()
