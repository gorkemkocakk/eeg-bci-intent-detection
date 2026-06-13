# parse_trials.py

import csv
import os
import sys

import config
from load_stieger import load_stieger_subject


# =========================================================
# 1) İLK DOĞRULAMA İÇİN BASİT AYARLAR
# =========================================================

# Şimdilik 1 session parse edeceğiz gibi sabit vermek yerine terminalden session alacağız.
# Böylece aynı kodu farklı session'lar için tekrar tekrar kullanabiliriz.
# Örnek:
# python parse_trials.py 1
# python parse_trials.py 2

# Parser: Verilen raw nesnesindeki annotation bilgisinden, her trial için ITI, cue ve feedback zamanlarını içeren bir tablo oluşturur.
# Bu tablo, daha sonra model eğitimi ve değerlendirmesi için kullanılabilir.

# Cue: Beyin aktivitesinin değiştiği an gibi düşünülebilir ama burada bizim için asıl önemli olan cue'nun başladığı zaman noktasını referans almak.
# Sonrasında ITI, cue ve feedback bölümlerini ayırıyoruz.

# Araştırmalara göre belirlenen trial yapısı:
# Stieger2021 dataset'inde her trial, 2 sn ITI + 2 sn cue + feedback süresinden oluşuyor.
# Bu süreler trial sınırlarını belirlemek için kullanılır.
ITI_DURATION_SEC = 2.0
CUE_DURATION_SEC = 2.0


# =========================================================
# 2) KÜÇÜK YARDIMCI FONKSİYONLAR
# =========================================================

def get_session_from_cli():
    """
    Terminalden session numarasını alır.

    Örnek:
    python parse_trials.py 1
    """
    if len(sys.argv) < 2:
        raise ValueError("Kullanım: python parse_trials.py <session_id>")

    session_name = str(sys.argv[1])
    return session_name


def sec_to_sample(sec, sfreq):
    """
    Saniyeyi sample indeksine çevirir. Örneğin, 2 saniye ve 250 Hz sampling rate için 500 sample eder.
    """
    return int(round(sec * sfreq))


def extract_triallength_sec(raw, annotation_index):
    """
    İlgili annotation satırından triallength bilgisini güvenli şekilde çeker.
    """
    if not hasattr(raw.annotations, "extras") or raw.annotations.extras is None:
        return None

    if annotation_index >= len(raw.annotations.extras):
        return None

    extra = raw.annotations.extras[annotation_index]

    if extra is None or "triallength" not in extra:
        return None

    try:
        triallength_sec = float(extra["triallength"])
    except (TypeError, ValueError):
        return None

    if triallength_sec <= 0:
        return None

    return triallength_sec


def get_session_raw(subject_data, session_name):
    """
    İstenen session içindeki ilk run'ın raw nesnesini döndürür.

    Not:
    - Bizim inventory çıktımıza göre her session'da 1 run vardı.
    - Bu yüzden ilk run'ı almak yeterli.
    """
    session_name = str(session_name)

    if session_name not in subject_data:
        raise ValueError(f"Session {session_name} bulunamadı.")

    runs = subject_data[session_name]
    run_names = list(runs.keys())

    if len(run_names) == 0:
        raise ValueError(f"Session {session_name} içinde run yok.")

    run_name = run_names[0]
    raw = runs[run_name]

    return raw, run_name


# =========================================================
# 3) ANNOTATION ÖZETİ
# =========================================================

# Annotation: EEG verisi üzerinde belirli olayların veya durumların işaretlendiği noktalardır.
# Örneğin bir deneme başlangıcı, bir uyarana tepki gibi olaylar annotation olarak kaydedilir.
# Annotation'lar, modelin öğrenmesi gereken kritik noktaları temsil eder.
# Bu noktaların zamanlaması, modelin performansını etkileyebilir.

def print_annotation_summary(raw, max_rows=10):
    """
    Annotation'ların kısa özetini ekrana basar.
    İlk iş olarak event yapısını gözle görmek için kullanılır.
    """
    if len(raw.annotations) == 0:
        raise ValueError("Bu raw nesnesinde annotation yok.")

    # np.str_ gibi görünmesin diye str'e çeviriyoruz
    unique_labels = sorted(list(set([str(x) for x in raw.annotations.description])))

    print("\n===== ANNOTATION ÖZETİ =====")
    print("Toplam annotation sayısı:", len(raw.annotations))
    print("Benzersiz annotation label'ları:", unique_labels)

    has_extras = hasattr(raw.annotations, "extras") and raw.annotations.extras is not None
    print("Triallength bilgisi var mı?:", has_extras)

    print("\nİlk birkaç annotation:")
    limit = min(max_rows, len(raw.annotations))

    for i in range(limit):
        onset_sec = round(float(raw.annotations.onset[i]), 3)
        label = str(raw.annotations.description[i])

        triallength_sec = None
        if has_extras and i < len(raw.annotations.extras):
            extra = raw.annotations.extras[i]
            if extra is not None and "triallength" in extra:
                triallength_sec = round(float(extra["triallength"]), 3)

        print(
            f"{i+1}. onset={onset_sec} sn | "
            f"label={label} | "
            f"triallength={triallength_sec}"
        )


# =========================================================
# 4) TRIAL PARSER
# =========================================================

def parse_trials_from_raw(raw, session_name, run_name):
    """
    Raw annotation bilgisinden trial tablosu üretir.

    Varsayım:
    - annotation onset = cue başlangıcı
    - cue süresi = 2 sn
    - feedback süresi = annotation extras içindeki triallength
    - feedback bitişi = feedback başlangıcı + triallength
    """
    if len(raw.annotations) == 0:
        raise ValueError("Trial parse etmek için yeterli annotation yok.")

    sfreq = float(raw.info["sfreq"])
    rows = []

    onsets = raw.annotations.onset
    labels = raw.annotations.description

    skipped_missing_triallength = 0

    for i in range(len(onsets)):
        current_onset = float(onsets[i])
        cue_label = str(labels[i])

        triallength_sec = extract_triallength_sec(raw, i)

        if triallength_sec is None:
            skipped_missing_triallength += 1
            print(
                f"Uyarı: Trial {i+1} için triallength bulunamadı/geçersiz. "
                f"Trial bilinçli şekilde atlandı."
            )
            continue

        iti_start_sec = max(0.0, current_onset - ITI_DURATION_SEC)
        iti_end_sec = current_onset

        cue_start_sec = current_onset
        cue_end_sec = cue_start_sec + CUE_DURATION_SEC

        feedback_start_sec = cue_end_sec
        feedback_end_sec = feedback_start_sec + triallength_sec

        trial_start_sec = iti_start_sec
        trial_end_sec = feedback_end_sec

        row = {
            "trial_id": i + 1,
            "session_id": str(session_name),
            "run_id": str(run_name),
            "cue_label": cue_label,

            "trial_start_sec": round(trial_start_sec, 4),
            "iti_start_sec": round(iti_start_sec, 4),
            "iti_end_sec": round(iti_end_sec, 4),
            "cue_start_sec": round(cue_start_sec, 4),
            "cue_end_sec": round(cue_end_sec, 4),
            "feedback_start_sec": round(feedback_start_sec, 4),
            "feedback_end_sec": round(feedback_end_sec, 4),
            "trial_end_sec": round(trial_end_sec, 4),

            "triallength_sec": round(triallength_sec, 4),
            "feedback_duration_sec": round(triallength_sec, 4),
            "source_trial_index": i,
            "feedback_semantic": "feedback_start_plus_triallength",

            "trial_start_sample": sec_to_sample(trial_start_sec, sfreq),
            "iti_start_sample": sec_to_sample(iti_start_sec, sfreq),
            "iti_end_sample": sec_to_sample(iti_end_sec, sfreq),
            "cue_start_sample": sec_to_sample(cue_start_sec, sfreq),
            "cue_end_sample": sec_to_sample(cue_end_sec, sfreq),
            "feedback_start_sample": sec_to_sample(feedback_start_sec, sfreq),
            "feedback_end_sample": sec_to_sample(feedback_end_sec, sfreq),
            "trial_end_sample": sec_to_sample(trial_end_sec, sfreq),
        }

        rows.append(row)

    if skipped_missing_triallength > 0:
        print(
            f"Bilgi: triallength eksik/geçersiz olduğu için atlanan trial sayısı: "
            f"{skipped_missing_triallength}"
        )

    return rows


# =========================================================
# 5) KISA ÖNİZLEME
# =========================================================

def print_first_trials(rows, max_rows=5):
    """
    Parse edilen ilk birkaç trial'ı ekrana basar.
    """
    print("\n===== İLK PARSE EDİLEN TRIAL'LAR =====")

    limit = min(max_rows, len(rows))

    for i in range(limit):
        row = rows[i]
        print(
            f"Trial {row['trial_id']} | "
            f"label={row['cue_label']} | "
            f"ITI=({row['iti_start_sec']}, {row['iti_end_sec']}) | "
            f"Cue=({row['cue_start_sec']}, {row['cue_end_sec']}) | "
            f"Feedback=({row['feedback_start_sec']}, {row['feedback_end_sec']})"
        )


# =========================================================
# 6) CSV KAYDI
# =========================================================

def save_trial_table(rows, session_name):
    """
    Trial tablosunu CSV olarak kaydeder.
    """
    if len(rows) == 0:
        raise ValueError("Kaydedilecek trial satırı yok.")

    save_dir = os.path.join(config.OUTPUT_DIR, "trial_tables")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_trial_table.csv")

    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTrial tablosu kaydedildi: {save_path}")


# =========================================================
# 7) ANA AKIŞ
# =========================================================

def main():
    subject_id = config.SUBJECT_ID
    session_name = get_session_from_cli()

    print(f"Subject {subject_id} yükleniyor...")
    subject_data = load_stieger_subject(subject_id)

    print(f"Session {session_name} seçiliyor...")
    raw, run_name = get_session_raw(subject_data, session_name)

    # Büyük sözlüğe artık ihtiyacımız yok
    del subject_data

    print_annotation_summary(raw)
    rows = parse_trials_from_raw(raw, session_name, run_name)

    print_first_trials(rows)
    save_trial_table(rows, session_name)

    print("\nTrial parsing tamamlandı.")


if __name__ == "__main__":
    main()
