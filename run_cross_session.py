# run_cross_session.py

import csv
import os
import re

import numpy as np

import config
from train_baseline import run_loso_baseline, print_final_summary, sort_session_ids


# =========================================================
# 1) KLASÖR YOLLARI
# =========================================================

FEATURE_FOLDER = os.path.join(config.OUTPUT_DIR, "features")
RESULTS_FOLDER = os.path.join(config.OUTPUT_DIR, "baseline_results")


# =========================================================
# 2) FEATURE DOSYALARINI BUL
# =========================================================

def get_session_feature_files():
    """
    outputs/features klasörü içindeki session feature dosyalarını bulur.

    Beklenen dosya adı örneği:
    session_1_bandpower_features.npz
    """
    if not os.path.exists(FEATURE_FOLDER):
        raise FileNotFoundError(f"Feature klasörü bulunamadı: {FEATURE_FOLDER}")

    session_files = {}

    for filename in os.listdir(FEATURE_FOLDER):
        if filename.endswith("_bandpower_features.npz") and filename.startswith("session_"):
            match = re.search(r"session_(\d+)_bandpower_features\.npz", filename)

            if match:
                session_id = match.group(1)
                full_path = os.path.join(FEATURE_FOLDER, filename)
                session_files[session_id] = full_path

    if len(session_files) == 0:
        raise ValueError("Hiç feature dosyası bulunamadı.")

    return session_files


# =========================================================
# 3) SESSION FEATURE YÜKLE
# =========================================================

def load_one_session_feature(file_path):
    """
    Tek bir session'ın feature dosyasını okur.

    Beklenen içerik:
    - X_features
    - y
    """
    data = np.load(file_path)

    X = data["X_features"]
    y = data["y"]

    return X, y


def load_all_sessions():
    """
    Tüm mevcut session feature dosyalarını yükler.

    Çıktı:
    session_data = {
        "1": {"X": ..., "y": ...},
        "2": {"X": ..., "y": ...},
        ...
    }
    """
    session_files = get_session_feature_files()
    sorted_ids = sort_session_ids(list(session_files.keys()))

    # Dosya adindan gelen session id'leri sayisal siraya dizilir, random shuffle yapilmaz.
    # Session-aware degerlendirme icin fold sirasi izlenebilir ve tekrar uretilebilir kalir.
    session_data = {}

    for session_id in sorted_ids:
        X, y = load_one_session_feature(session_files[session_id])

        session_data[session_id] = {
            "X": X,
            "y": y
        }

    return session_data


# =========================================================
# 4) SONUÇLARI KAYDET
# =========================================================

def save_fold_results(fold_results):
    """
    Fold sonuçlarını CSV olarak kaydeder.
    """
    # Fold CSV, raporda kullanilacak ana metrikleri ve hangi session'in testte kaldigini saklar.
    # Bu dosya daha sonra sonuc karsilastirmalari icin tekrar okunabilir.
    if len(fold_results) == 0:
        raise ValueError("Kaydedilecek fold sonucu yok.")

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    save_path = os.path.join(RESULTS_FOLDER, "loso_bandpower_lda_results.csv")

    fieldnames = list(fold_results[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_results)

    print(f"\nFold sonuçları kaydedildi: {save_path}")


def save_predictions(all_predictions):
    """
    Test fold tahminlerini CSV olarak kaydeder.
    """
    # Tahmin CSV'si pencere bazinda true/pred/score saklar.
    # Yanlis siniflanan ITI veya feedback pencerelerini sonradan incelemek icin gereklidir.
    if len(all_predictions) == 0:
        raise ValueError("Kaydedilecek tahmin yok.")

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    save_path = os.path.join(RESULTS_FOLDER, "loso_bandpower_lda_predictions.csv")

    fieldnames = list(all_predictions[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_predictions)

    print(f"Tahminler kaydedildi: {save_path}")


# =========================================================
# 5) ANA AKIŞ
# =========================================================

def main():
    print("Session feature dosyaları aranıyor...")
    session_data = load_all_sessions()

    session_ids = sort_session_ids(list(session_data.keys()))
    print("Bulunan session'lar:", session_ids)

    # Bu dosya tüm mevcut session feature'larını alır ve cross-session baseline'ı başlatır.
    # Yani tek tek session vermiyoruz, klasörde ne varsa onunla koşuyoruz.
    fold_results, all_predictions = run_loso_baseline(session_data, verbose=True)

    print_final_summary(fold_results)
    save_fold_results(fold_results)
    save_predictions(all_predictions)

    print("\nCross-session koşusu tamamlandı.")


if __name__ == "__main__":
    main()