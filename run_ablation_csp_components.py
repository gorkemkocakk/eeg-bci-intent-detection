import argparse
import csv
import os

import numpy as np

import config
from experiment_manifest import write_manifest
from run_cross_session_csp_5band import load_all_sessions
from train_csp_5band_baseline import run_loso_csp_5band_baseline


# Bu script CSP component sayisinin LOSO-session performansina etkisini inceler.
# Ablation ciktisi model secimi kaniti degil, rapor icin kontrollu karsilastirma tablosudur.
RESULTS_FOLDER = os.path.join(config.OUTPUT_DIR, "ablation_results")
COMPONENT_VALUES = [2, 4, 6, 8]
WIDEBAND_WINDOW_FOLDER = os.path.join(config.OUTPUT_DIR, "window_data_wideband")
EXPECTED_WIDEBAND_WINDOW_COUNT = 11


def preflight_wideband_inputs():
    print("Current working directory:", os.getcwd())

    # 5-band CSP ablation, tum session'larin ayni wideband pencere setinden gelmesini bekler.
    # Eksik dosya varsa fold sayisi degisir ve component karsilastirmasi adil olmaz.
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


def parse_args():
    # CLI, yalnizca session araligini ve component listesini degistirir.
    # Egitim mantigi ve LOSO-session ayrimi training fonksiyonlarinda sabit kalir.
    parser = argparse.ArgumentParser(
        description="Run LOSO CSP component ablations on existing wideband windows."
    )
    parser.add_argument(
        "session_range",
        nargs="*",
        type=int,
        metavar="SESSION",
        help="Optional start and end session, for example: 1 3",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        type=int,
        help="Positive CSP component values to run, for example: --components 4 8",
    )
    args = parser.parse_args()

    if len(args.session_range) not in (0, 2):
        parser.error("Provide either no session range or exactly two values: start_session end_session")

    if len(args.session_range) == 2:
        start_session, end_session = args.session_range
        if start_session > end_session:
            parser.error("start_session cannot be greater than end_session")
    else:
        start_session, end_session = None, None

    if args.components is None:
        components = COMPONENT_VALUES
    else:
        invalid_components = [value for value in args.components if value <= 0]
        if invalid_components:
            parser.error(
                "--components values must be positive integers. "
                f"Invalid values: {invalid_components}"
            )
        components = args.components

    return start_session, end_session, components


def save_csv(rows, save_path):
    # Her component degeri icin fold sonuclari, sonunda da ozet tablo kaydedilir.
    # Sabit CSV isimleri sonraki summarize ve istatistik scriptlerinin girdisidir.
    if len(rows) == 0:
        raise ValueError(f"Kaydedilecek satir yok: {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Kaydedildi: {save_path}")


def build_summary_row(csp_components, fold_results):
    auc_values = [float(row["roc_auc"]) for row in fold_results]
    bal_values = [float(row["balanced_accuracy"]) for row in fold_results]

    return {
        "csp_components": int(csp_components),
        "mean_auc": round(float(np.mean(auc_values)), 6),
        "std_auc": round(float(np.std(auc_values)), 6),
        "mean_balanced_accuracy": round(float(np.mean(bal_values)), 6),
        "std_balanced_accuracy": round(float(np.std(bal_values)), 6),
        "fold_count": int(len(fold_results)),
    }


def main():
    start_session, end_session, component_values = parse_args()
    preflight_wideband_inputs()
    write_manifest(
        RESULTS_FOLDER,
        script_name=os.path.basename(__file__),
        cli_args={
            "start_session": start_session,
            "end_session": end_session,
            "components": component_values,
        },
        input_dir=WIDEBAND_WINDOW_FOLDER,
        output_dir_value=RESULTS_FOLDER,
        extra={
            "evaluation": "LOSO-session",
            "model_family": "5-band CSP + LDA",
        },
    )

    print("Session wideband window dosyalari yukleniyor...")
    session_data = load_all_sessions(start_session, end_session)

    summary_rows = []

    # Component sayisi degisse bile train/test session ayrimi ayni kalir.
    # Boylece metrik farklari split degisiminden degil CSP ayarindan kaynaklanir.
    for csp_components in component_values:
        print(f"\n===== CSP COMPONENT ABLATION: {csp_components} =====")

        fold_results, _ = run_loso_csp_5band_baseline(
            session_data=session_data,
            verbose=True,
            csp_components=csp_components,
        )

        per_value_path = os.path.join(
            RESULTS_FOLDER,
            f"csp_components_{csp_components}_loso_results.csv",
        )
        save_csv(fold_results, per_value_path)

        summary_rows.append(build_summary_row(csp_components, fold_results))

    summary_path = os.path.join(RESULTS_FOLDER, "csp_components_ablation_summary.csv")
    save_csv(summary_rows, summary_path)

    print("\nCSP components ablation tamamlandi.")


if __name__ == "__main__":
    main()
