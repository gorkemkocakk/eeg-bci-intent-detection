# run_cross_session_csp.py

import csv
import os
import re
import sys

import numpy as np

import config
from train_csp_baseline import run_loso_csp_baseline, print_final_summary, sort_session_ids


# =========================================================
# 1) KLASOR YOLLARI
# =========================================================

WINDOW_FOLDER = os.path.join(config.OUTPUT_DIR, "window_data")
RESULTS_FOLDER = os.path.join(config.OUTPUT_DIR, "baseline_results")


# =========================================================
# 2) BASIT CLI
# =========================================================

def get_session_range_from_cli():
    """
    Opsiyonel session araligini alir.

    Kullanim:
    python run_cross_session_csp.py
    python run_cross_session_csp.py 1 3
    """
    if len(sys.argv) == 1:
        return None, None

    if len(sys.argv) == 3:
        start_session = int(sys.argv[1])
        end_session = int(sys.argv[2])

        # Aralik secimi sadece smoke/alt deney icindir; secilen session'lar yine sirali ve shuffle'siz kalir.
        if start_session > end_session:
            raise ValueError("Baslangic session, bitis session'dan buyuk olamaz.")

        return start_session, end_session

    raise ValueError("Kullanim: python run_cross_session_csp.py [start_session end_session]")


# =========================================================
# 3) WINDOW DOSYALARINI BUL
# =========================================================

def get_session_window_files():
    """
    outputs/window_data klasoru icindeki session window dosyalarini bulur.

    Beklenen dosya adi ornegi:
    session_1_windows.npz
    """
    if not os.path.exists(WINDOW_FOLDER):
        raise FileNotFoundError(f"Window klasoru bulunamadi: {WINDOW_FOLDER}")

    session_files = {}

    for filename in os.listdir(WINDOW_FOLDER):
        if filename.endswith("_windows.npz") and filename.startswith("session_"):
            match = re.search(r"session_(\d+)_windows\.npz", filename)

            if match:
                session_id = match.group(1)
                full_path = os.path.join(WINDOW_FOLDER, filename)
                session_files[session_id] = full_path

    if len(session_files) == 0:
        raise ValueError("Hic window dosyasi bulunamadi.")

    return session_files


# =========================================================
# 4) SESSION WINDOW YUKLE
# =========================================================

def load_one_session_window(file_path):
    """
    Tek bir session'in window dosyasini okur.

    Beklenen icerik:
    - X -> (n_windows, n_channels, n_samples)
    - y -> (n_windows,)
    """
    data = np.load(file_path)

    X = data["X"]
    y = data["y"]

    return X, y


def load_all_sessions(start_session=None, end_session=None):
    """
    Tum mevcut session window dosyalarini yukler.

    Cikti:
    session_data = {
        "1": {"X": ..., "y": ...},
        "2": {"X": ..., "y": ...},
        ...
    }
    """
    session_files = get_session_window_files()
    sorted_ids = sort_session_ids(list(session_files.keys()))

    # CSP feature'lari burada onceden hesaplanmaz.
    # Her fold kendi train session'lariyla CSP fit edecegi icin ham window X/y yuklenir.
    if start_session is not None and end_session is not None:
        sorted_ids = [
            session_id for session_id in sorted_ids
            if start_session <= int(session_id) <= end_session
        ]

        if len(sorted_ids) == 0:
            raise ValueError("Verilen aralikta session window dosyasi bulunamadi.")

    session_data = {}

    for session_id in sorted_ids:
        X, y = load_one_session_window(session_files[session_id])

        session_data[session_id] = {
            "X": X,
            "y": y
        }

    return session_data


# =========================================================
# 5) SONUCLARI KAYDET
# =========================================================

def save_fold_results(fold_results):
    """
    Fold sonuclarini CSV olarak kaydeder.
    """
    # Sonuc dosyasi, LOSO fold'larinin metriklerini sabit dosya adiyla saklar.
    # Raporlama ve tekrar kontrol icin train/test session bilgisi satirda tutulur.
    if len(fold_results) == 0:
        raise ValueError("Kaydedilecek fold sonucu yok.")

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    save_path = os.path.join(RESULTS_FOLDER, "loso_csp_lda_results.csv")

    fieldnames = list(fold_results[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_results)

    print(f"\nFold sonuclari kaydedildi: {save_path}")


def save_predictions(all_predictions):
    """
    Test fold tahminlerini CSV olarak kaydeder.
    """
    # Prediction dosyasi pencere bazinda skor saklar; ROC-AUC ve hata analizi buradan izlenebilir.
    if len(all_predictions) == 0:
        raise ValueError("Kaydedilecek tahmin yok.")

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    save_path = os.path.join(RESULTS_FOLDER, "loso_csp_lda_predictions.csv")

    fieldnames = list(all_predictions[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_predictions)

    print(f"Tahminler kaydedildi: {save_path}")


# =========================================================
# 6) ANA AKIS
# =========================================================

def main():
    start_session, end_session = get_session_range_from_cli()

    print("Session window dosyalari araniyor...")
    session_data = load_all_sessions(start_session, end_session)

    session_ids = sort_session_ids(list(session_data.keys()))
    print("Bulunan session'lar:", session_ids)

    # Bu dosya tum mevcut session window'larini alir ve cross-session CSP baseline'i baslatir.
    # Yani session bazli CSP precompute yapilmaz; CSP fold icinde train veride fit edilir.
    fold_results, all_predictions = run_loso_csp_baseline(session_data, verbose=True)

    print_final_summary(fold_results)
    save_fold_results(fold_results)
    save_predictions(all_predictions)

    print("\nCross-session CSP kosusu tamamlandi.")


if __name__ == "__main__":
    main()