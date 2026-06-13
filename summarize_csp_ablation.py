import argparse
import csv
from pathlib import Path
import re
import sys

import numpy as np


RESULTS_DIR = Path("outputs") / "ablation_results"
INPUT_PATTERN = "csp_components_*_loso_results.csv"
REQUIRED_COLUMNS = {"test_session", "roc_auc", "balanced_accuracy"}


# Bu script yeni model egitmez; mevcut component ablation CSV'lerini ozetler.
# Ayrica 4 ve 8 component sonuclarini ayni test session bazinda eslestirerek karsilastirir.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize existing CSP component ablation CSV files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing csp_components_*_loso_results.csv files.",
    )
    return parser.parse_args()


def read_rows(path):
    # Girdi CSV'lerinde test_session ve ana metrikler yoksa ozet yanlis anlam kazanir.
    # Bu kontrol, ablation ciktisi yerine baska bir CSV'nin yanlislikla verilmesini erken yakalar.
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required columns: {missing_text}")
        return list(reader)


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csp_components_from_name(path):
    # Component sayisi dosya adindan parse edilir; satir icindeki metriklerle karistirilmaz.
    # Sabit adlandirma, ablation klasorundeki dosyalari otomatik toplamayi saglar.
    match = re.fullmatch(r"csp_components_(\d+)_loso_results\.csv", path.name)
    if not match:
        raise ValueError(f"Could not parse CSP component count from {path.name}")
    return int(match.group(1))


def as_float(row, column, path):
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"{path} has non-numeric {column!r}: {row[column]!r}") from exc


def summarize_file(path):
    rows = read_rows(path)
    auc_values = np.array([as_float(row, "roc_auc", path) for row in rows], dtype=float)
    ba_values = np.array(
        [as_float(row, "balanced_accuracy", path) for row in rows],
        dtype=float,
    )

    return {
        "csp_components": csp_components_from_name(path),
        "mean_auc": round(float(np.mean(auc_values)), 6),
        "std_auc": round(float(np.std(auc_values)), 6),
        "mean_balanced_accuracy": round(float(np.mean(ba_values)), 6),
        "std_balanced_accuracy": round(float(np.std(ba_values)), 6),
        "fold_count": len(rows),
    }


def print_summary(rows):
    fieldnames = [
        "csp_components",
        "mean_auc",
        "std_auc",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "fold_count",
    ]
    widths = {
        field: max(len(field), *(len(str(row[field])) for row in rows))
        for field in fieldnames
    }

    print("CSP component ablation summary:")
    print("  ".join(field.rjust(widths[field]) for field in fieldnames))
    print("  ".join("-" * widths[field] for field in fieldnames))
    for row in rows:
        print("  ".join(str(row[field]).rjust(widths[field]) for field in fieldnames))


def rows_by_test_session(path):
    rows = read_rows(path)
    keyed = {}
    for row in rows:
        test_session = row["test_session"]
        if test_session in keyed:
            raise ValueError(f"{path} has duplicate test_session: {test_session}")
        keyed[test_session] = row
    return keyed


def build_fold_diff(path_c4, path_c8):
    # Fold bazli fark hesabi icin iki dosyanin ayni test_session setini icermesi gerekir.
    # Aksi halde ortalama fark, farkli session'lardan gelen skorlarin karisimi olur.
    c4_rows = rows_by_test_session(path_c4)
    c8_rows = rows_by_test_session(path_c8)

    if set(c4_rows) != set(c8_rows):
        missing_from_c8 = sorted(set(c4_rows) - set(c8_rows), key=int)
        missing_from_c4 = sorted(set(c8_rows) - set(c4_rows), key=int)
        raise ValueError(
            "4-vs-8 comparison requires matching test_session values. "
            f"Missing from c8: {missing_from_c8}; missing from c4: {missing_from_c4}"
        )

    diff_rows = []
    for test_session in sorted(c4_rows, key=int):
        c4 = c4_rows[test_session]
        c8 = c8_rows[test_session]
        auc_c4 = as_float(c4, "roc_auc", path_c4)
        ba_c4 = as_float(c4, "balanced_accuracy", path_c4)
        auc_c8 = as_float(c8, "roc_auc", path_c8)
        ba_c8 = as_float(c8, "balanced_accuracy", path_c8)
        delta_auc = auc_c8 - auc_c4
        delta_ba = ba_c8 - ba_c4

        diff_rows.append(
            {
                "test_session": test_session,
                "roc_auc_c4": round(auc_c4, 6),
                "balanced_accuracy_c4": round(ba_c4, 6),
                "roc_auc_c8": round(auc_c8, 6),
                "balanced_accuracy_c8": round(ba_c8, 6),
                "delta_auc_8_minus_4": round(delta_auc, 6),
                "delta_ba_8_minus_4": round(delta_ba, 6),
            }
        )

    return diff_rows


def print_fold_diff_stats(diff_rows):
    delta_auc = np.array(
        [float(row["delta_auc_8_minus_4"]) for row in diff_rows],
        dtype=float,
    )
    delta_ba = np.array(
        [float(row["delta_ba_8_minus_4"]) for row in diff_rows],
        dtype=float,
    )

    print()
    print("4-vs-8 fold-level comparison:")
    print("average delta AUC:", round(float(np.mean(delta_auc)), 6))
    print("average delta Balanced Accuracy:", round(float(np.mean(delta_ba)), 6))
    print("folds where 8 beats 4 in ROC-AUC:", int(np.sum(delta_auc > 0)))
    print("folds where 8 beats 4 in Balanced Accuracy:", int(np.sum(delta_ba > 0)))


def main():
    args = parse_args()
    results_dir = args.results_dir
    summary_path = results_dir / "csp_components_ablation_summary.csv"
    fold_diff_path = results_dir / "csp_components_4_vs_8_fold_diff.csv"

    print("Input/output directory:", results_dir.resolve())

    input_files = sorted(
        path for path in results_dir.glob(INPUT_PATTERN)
        if "summary" not in path.name
    )
    # Summary dosyalari tekrar summary girdisi yapilmaz.
    # Sadece per-component LOSO sonuc CSV'leri bu ozetin kaynagi olmalidir.
    print("Matched input files:")
    for path in input_files:
        print(f"  {path.resolve()}")

    if not input_files:
        print(f"ERROR: No files found matching {results_dir / INPUT_PATTERN}")
        sys.exit(1)

    try:
        summary_rows = [summarize_file(path) for path in input_files]
        summary_rows.sort(key=lambda row: row["mean_auc"], reverse=True)

        summary_fieldnames = [
            "csp_components",
            "mean_auc",
            "std_auc",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
            "fold_count",
        ]
        write_rows(summary_path, summary_rows, summary_fieldnames)
        print_summary(summary_rows)
        print()
        print(f"Saved summary: {summary_path}")

        path_c4 = results_dir / "csp_components_4_loso_results.csv"
        path_c8 = results_dir / "csp_components_8_loso_results.csv"
        if path_c4.is_file() and path_c8.is_file():
            diff_rows = build_fold_diff(path_c4, path_c8)
            diff_fieldnames = [
                "test_session",
                "roc_auc_c4",
                "balanced_accuracy_c4",
                "roc_auc_c8",
                "balanced_accuracy_c8",
                "delta_auc_8_minus_4",
                "delta_ba_8_minus_4",
            ]
            write_rows(fold_diff_path, diff_rows, diff_fieldnames)
            print(f"Saved fold comparison: {fold_diff_path}")
            print_fold_diff_stats(diff_rows)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
