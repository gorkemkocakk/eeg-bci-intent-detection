# build_labels.py

import csv
import os
import sys

import config


# =========================================================
# 1) İLK DOĞRULAMA İÇİN BASİT AYAR
# =========================================================

# Şimdilik sadece parser'ını doğruladığımız session üzerinde çalışıyoruz
# demek yerine session bilgisini terminalden alacağız.
# Böylece aynı kodu bütün session'lar için tekrar tekrar kullanabiliriz.
# Örnek:
# python build_labels.py 1
# python build_labels.py 2


# =========================================================
# 2) DOSYA OKUMA
# =========================================================

def get_session_from_cli():
    """
    Terminalden session numarasını alır.

    Örnek:
    python build_labels.py 1
    """
    if len(sys.argv) < 2:
        raise ValueError("Kullanım: python build_labels.py <session_id>")

    session_name = str(sys.argv[1])
    return session_name


def read_trial_table(session_name):
    """
    Parser'ın ürettiği trial tablosunu okur.

    Girdi:
        session_name -> örn: "1"

    Çıktı:
        trial_rows -> her satırı sözlük olan liste

    Not:
    Bu dosya ham EEG okumaz.
    Sadece parse_trials.py çıktısını kullanır.
    """
    file_path = os.path.join(
        config.OUTPUT_DIR,
        "trial_tables",
        f"session_{session_name}_trial_table.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trial tablosu bulunamadı: {file_path}")

    trial_rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_rows.append(row)

    if len(trial_rows) == 0:
        raise ValueError("Trial tablosu boş görünüyor.")

    return trial_rows


# =========================================================
# 3) LABEL TABLOSU ÜRETME
# =========================================================

def build_label_rows(trial_rows):
    """
    Trial tablosundan etiket tablosu üretir.

    Kural:
    - ITI segmenti -> label 0
    - Feedback segmenti -> label 1
    - Cue segmenti tabloya alınmaz

    Çıktı:
        label_rows -> her satır bir segment olacak şekilde liste
    """
    label_rows = []
    segment_id = 1

    for trial in trial_rows:
        trial_id = int(trial["trial_id"])
        session_id = str(trial["session_id"])
        run_id = str(trial["run_id"])
        cue_label = str(trial["cue_label"])

        # Bu projede 0 sinifi yalnizca ITI'den gelir.
        # Cue komutu veya voluntary rest/down bilgisi idle/non-control etiketi sayilmaz.
        # -------------------------------------------------
        # ITI segmenti -> 0
        # -------------------------------------------------
        iti_row = {
            "segment_id": segment_id,
            "trial_id": trial_id,
            "session_id": session_id,
            "run_id": run_id,
            "source_segment": "ITI",
            "label": 0,
            "cue_label": cue_label,

            "start_sec": float(trial["iti_start_sec"]),
            "end_sec": float(trial["iti_end_sec"]),
            "duration_sec": round(
                float(trial["iti_end_sec"]) - float(trial["iti_start_sec"]), 4
            ),

            "start_sample": int(trial["iti_start_sample"]),
            "end_sample": int(trial["iti_end_sample"]),
        }

        label_rows.append(iti_row)
        segment_id += 1

        # Feedback segmenti 1 sinifini temsil eder.
        # Kontrol var/yok ayriminda cue_label hedef yonu olarak saklanir, sinif etiketi olmaz.
        # -------------------------------------------------
        # Feedback segmenti -> 1
        # -------------------------------------------------
        feedback_row = {
            "segment_id": segment_id,
            "trial_id": trial_id,
            "session_id": session_id,
            "run_id": run_id,
            "source_segment": "feedback",
            "label": 1,
            "cue_label": cue_label,

            "start_sec": float(trial["feedback_start_sec"]),
            "end_sec": float(trial["feedback_end_sec"]),
            "duration_sec": round(
                float(trial["feedback_end_sec"]) - float(trial["feedback_start_sec"]), 4
            ),

            "start_sample": int(trial["feedback_start_sample"]),
            "end_sample": int(trial["feedback_end_sample"]),
        }

        label_rows.append(feedback_row)
        segment_id += 1

    return label_rows


# =========================================================
# 4) KONTROL AMAÇLI ÖZET
# =========================================================

def print_label_preview(label_rows, max_rows=10):
    """
    İlk birkaç etiket satırını ekrana basar.
    """
    print("\n===== İLK LABEL SATIRLARI =====")

    limit = min(max_rows, len(label_rows))

    for i in range(limit):
        row = label_rows[i]
        print(
            f"segment_id={row['segment_id']} | "
            f"trial_id={row['trial_id']} | "
            f"type={row['source_segment']} | "
            f"label={row['label']} | "
            f"start={row['start_sec']} | "
            f"end={row['end_sec']} | "
            f"duration={row['duration_sec']}"
        )


def print_label_summary(label_rows):
    """
    Üretilen label tablosunun kısa özetini verir.
    """
    n_iti = 0
    n_feedback = 0

    total_iti_sec = 0.0
    total_feedback_sec = 0.0

    for row in label_rows:
        if row["label"] == 0:
            n_iti += 1
            total_iti_sec += row["duration_sec"]
        elif row["label"] == 1:
            n_feedback += 1
            total_feedback_sec += row["duration_sec"]

    print("\n===== LABEL ÖZETİ =====")
    print("Toplam segment sayısı:", len(label_rows))
    print("ITI segment sayısı (label 0):", n_iti)
    print("Feedback segment sayısı (label 1):", n_feedback)
    print("Toplam ITI süresi (sn):", round(total_iti_sec, 2))
    print("Toplam Feedback süresi (sn):", round(total_feedback_sec, 2))


# =========================================================
# 5) CSV KAYDI
# =========================================================

def save_label_table(label_rows, session_name):
    """
    Label tablosunu CSV olarak kaydeder.
    """
    if len(label_rows) == 0:
        raise ValueError("Kaydedilecek label satırı yok.")

    save_dir = os.path.join(config.OUTPUT_DIR, "label_tables")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_labels.csv")

    fieldnames = list(label_rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"\nLabel tablosu kaydedildi: {save_path}")


# =========================================================
# 6) ANA AKIŞ
# =========================================================

def main():
    session_name = get_session_from_cli()

    print(f"Session {session_name} için trial tablosu okunuyor...")
    trial_rows = read_trial_table(session_name)

    print("Label tablosu üretiliyor...")
    label_rows = build_label_rows(trial_rows)

    print_label_preview(label_rows)
    print_label_summary(label_rows)
    save_label_table(label_rows, session_name)

    print("\nEtiketleme tamamlandı.")


if __name__ == "__main__":
    main()