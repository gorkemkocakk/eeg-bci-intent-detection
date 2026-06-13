import csv
import os
import sys

import numpy as np

import config
from run_cross_session_csp_5band_fs import load_all_sessions
from train_csp_5band_fs_baseline import run_loso_csp_5band_fs_baseline


# Bu script, 5-band CSP sonrasi feature selection k degerinin etkisini karsilastirir.
# K secimi burada post-hoc ablation amaclidir; nihai tarafsiz secim icin nested LOSO gerekir.
RESULTS_FOLDER = os.path.join(config.OUTPUT_DIR, "ablation_results")
K_VALUES = [5, 10, 15, 20, "all"]
WIDEBAND_WINDOW_FOLDER = os.path.join(config.OUTPUT_DIR, "window_data_wideband")
EXPECTED_WIDEBAND_WINDOW_COUNT = 11


def preflight_wideband_inputs():
    print("Current working directory:", os.getcwd())

    # Ablation baslamadan once tum wideband pencere dosyalarinin hazir oldugu dogrulanir.
    # Eksik session ile kosmak fold sayisini degistirir ve sonuclari raporla karsilastirmayi bozar.
    if not os.path.isdir(WIDEBAND_WINDOW_FOLDER):
        found_count = 0
    else:
        found_count = len(
            [
                filename for filename in os.listdir(WIDEBAND_WINDOW_FOLDER)
                if filename.startswith("session_")
                and filename.endswith("_wideband_windows.npz")
            ]
        )

    print("Wideband window file count:", found_count)

    if found_count != EXPECTED_WIDEBAND_WINDOW_COUNT:
        raise RuntimeError(
            f"Expected 11 wideband window files before ablation, found {found_count}. "
            "Run windowing_wideband.py for sessions 1..11 first."
        )


def get_session_range_from_cli():
    """
    Opsiyonel session araligini alir.

    Kullanim:
    python run_ablation_fs_k.py
    python run_ablation_fs_k.py 1 3
    """
    if len(sys.argv) == 1:
        return None, None

    if len(sys.argv) == 3:
        start_session = int(sys.argv[1])
        end_session = int(sys.argv[2])

        if start_session > end_session:
            raise ValueError("Baslangic session, bitis session'dan buyuk olamaz.")

        return start_session, end_session

    raise ValueError(
        "Kullanim: python run_ablation_fs_k.py [start_session end_session]"
    )


def save_csv(rows, save_path):
    # Her k degeri ve ozet ayri CSV'ye yazilir.
    # Bu cikti bir sonraki analizlerde tekrar okunacak deney artefaktidir.
    if len(rows) == 0:
        raise ValueError(f"Kaydedilecek satir yok: {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Kaydedildi: {save_path}")


def build_summary_row(k_value, fold_results):
    auc_values = [float(row["roc_auc"]) for row in fold_results]
    bal_values = [float(row["balanced_accuracy"]) for row in fold_results]

    return {
        "k": str(k_value),
        "mean_auc": round(float(np.mean(auc_values)), 6),
        "std_auc": round(float(np.std(auc_values)), 6),
        "mean_balanced_accuracy": round(float(np.mean(bal_values)), 6),
        "std_balanced_accuracy": round(float(np.std(bal_values)), 6),
        "fold_count": int(len(fold_results)),
    }


def main():
    preflight_wideband_inputs()
    start_session, end_session = get_session_range_from_cli()

    print("Session wideband window dosyalari yukleniyor...")
    session_data = load_all_sessions(start_session, end_session)

    summary_rows = []

    # Ayni session_data uzerinde k degeri degistirilir.
    # Feature selection fit'i training fonksiyonunda her LOSO fold'unun train tarafinda kalir.
    for k_value in K_VALUES:
        print(f"\n===== FS k ABLATION: {k_value} =====")

        fold_results, _ = run_loso_csp_5band_fs_baseline(
            session_data=session_data,
            verbose=True,
            csp_components=int(config.CSP_COMPONENTS),
            k_best=k_value,
        )

        annotated_rows = []
        for row in fold_results:
            new_row = dict(row)
            new_row["k_requested"] = str(k_value)
            annotated_rows.append(new_row)

        per_value_path = os.path.join(
            RESULTS_FOLDER,
            f"fs_k_{k_value}_loso_results.csv",
        )
        save_csv(annotated_rows, per_value_path)

        summary_rows.append(build_summary_row(k_value, fold_results))

    summary_path = os.path.join(RESULTS_FOLDER, "fs_k_ablation_summary.csv")
    save_csv(summary_rows, summary_path)

    print("\nFS k ablation tamamlandi.")


if __name__ == "__main__":
    main()
