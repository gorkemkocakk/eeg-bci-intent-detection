import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from experiment_manifest import write_manifest


DEFAULT_RESULTS_DIR = Path("outputs") / "ablation_results"
SESSION_COLUMNS = ["test_session", "session", "session_id", "fold", "fold_id"]
ROC_AUC_COLUMNS = ["roc_auc", "auc", "mean_roc_auc"]
BALANCED_ACCURACY_COLUMNS = ["balanced_accuracy", "bal_acc", "ba"]


# Bu script iki CSP component ayarini fold bazinda eslestirerek post-hoc istatistik uretir.
# Yeni egitim yapmaz; mevcut ablation CSV'lerini okur ve kucuk n nedeniyle sonuclari kesifsel yorumlar.
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired fold-level diagnostics for two CSP component "
            "ablation result CSVs."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--component-a", type=int, default=4)
    parser.add_argument("--component-b", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        default="csp_components_4_vs_8_stats",
        help="Prefix for paired delta, summary CSV, and summary JSON outputs.",
    )
    args = parser.parse_args()

    if args.component_a <= 0 or args.component_b <= 0:
        parser.error("--component-a and --component-b must be positive integers")
    if args.component_a == args.component_b:
        parser.error("--component-a and --component-b must be different")
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be a positive integer")

    return args


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def detect_column(fieldnames, candidates, label, path):
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"{path} does not contain a {label} column. "
        f"Accepted names: {', '.join(candidates)}"
    )


def as_float(row, column, path):
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"{path} has non-numeric {column!r}: {row[column]!r}") from exc


def load_component_results(path):
    # Farkli CSV surumlerinde kolon isimleri degisebildigi icin kabul edilen isimlerden biri aranir.
    # Fold/session anahtari benzersiz olmazsa paired analiz anlamini kaybeder.
    rows, fieldnames = read_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    session_column = detect_column(fieldnames, SESSION_COLUMNS, "session/fold id", path)
    auc_column = detect_column(fieldnames, ROC_AUC_COLUMNS, "ROC-AUC metric", path)
    ba_column = detect_column(
        fieldnames,
        BALANCED_ACCURACY_COLUMNS,
        "Balanced Accuracy metric",
        path,
    )

    keyed = {}
    for row in rows:
        session_id = row[session_column]
        if session_id in keyed:
            raise ValueError(f"{path} has duplicate session/fold id: {session_id}")
        keyed[session_id] = {
            "session_id": session_id,
            "roc_auc": as_float(row, auc_column, path),
            "balanced_accuracy": as_float(row, ba_column, path),
        }

    return keyed, {
        "session_column": session_column,
        "roc_auc_column": auc_column,
        "balanced_accuracy_column": ba_column,
    }


def sort_session_ids(session_ids):
    def sort_key(value):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(session_ids, key=sort_key)


def build_paired_delta_rows(component_a, component_b, rows_a, rows_b):
    # Paired delta, ayni test session icin component B skorundan component A skorunu cikarir.
    # Session setleri eslesmezse farklar genelleme degil dosya uyumsuzlugu olur.
    if set(rows_a) != set(rows_b):
        missing_from_b = sort_session_ids(set(rows_a) - set(rows_b))
        missing_from_a = sort_session_ids(set(rows_b) - set(rows_a))
        raise ValueError(
            "Component result files must contain matching session/fold ids. "
            f"Missing from component B: {missing_from_b}; "
            f"missing from component A: {missing_from_a}"
        )

    paired_rows = []
    for session_id in sort_session_ids(rows_a.keys()):
        row_a = rows_a[session_id]
        row_b = rows_b[session_id]
        auc_delta = row_b["roc_auc"] - row_a["roc_auc"]
        ba_delta = row_b["balanced_accuracy"] - row_a["balanced_accuracy"]
        paired_rows.append(
            {
                "session_id": session_id,
                "component_a": component_a,
                "component_b": component_b,
                "roc_auc_a": round(row_a["roc_auc"], 6),
                "roc_auc_b": round(row_b["roc_auc"], 6),
                "delta_roc_auc_b_minus_a": round(float(auc_delta), 6),
                "balanced_accuracy_a": round(row_a["balanced_accuracy"], 6),
                "balanced_accuracy_b": round(row_b["balanced_accuracy"], 6),
                "delta_balanced_accuracy_b_minus_a": round(float(ba_delta), 6),
            }
        )
    return paired_rows


def wilcoxon_p_value(deltas):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("WARNING: scipy is unavailable; Wilcoxon p-value will be null.")
        return None

    if np.allclose(deltas, 0):
        return 1.0

    try:
        result = wilcoxon(deltas, alternative="two-sided")
    except ValueError as exc:
        print(f"WARNING: Wilcoxon test could not be computed: {exc}")
        return None
    return float(result.pvalue)


def bootstrap_mean_ci(deltas, n_bootstrap, seed):
    # Bootstrap burada fold-level ortalama fark icin belirsizlik araligi verir.
    # Bu, nested model secimi yerine gecmez; yalnizca mevcut fold farklarini ozetler.
    rng = np.random.default_rng(seed)
    n = len(deltas)
    samples = rng.choice(deltas, size=(n_bootstrap, n), replace=True)
    means = np.mean(samples, axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def summarize_metric(metric_name, deltas, n_bootstrap, seed):
    n = int(len(deltas))
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap, seed)
    p_value = wilcoxon_p_value(deltas)

    if std_delta == 0:
        cohen_dz = None
    else:
        cohen_dz = mean_delta / std_delta

    tol = 1e-12
    greater = int(np.sum(deltas > tol))
    equal = int(np.sum(np.abs(deltas) <= tol))
    less = int(np.sum(deltas < -tol))

    return {
        "metric": metric_name,
        "n": n,
        "mean_delta_b_minus_a": round(mean_delta, 6),
        "std_delta": round(std_delta, 6),
        "median_delta": round(float(np.median(deltas)), 6),
        "min_delta": round(float(np.min(deltas)), 6),
        "max_delta": round(float(np.max(deltas)), 6),
        "folds_b_greater_a": greater,
        "folds_b_equal_a": equal,
        "folds_b_less_a": less,
        "percent_folds_b_greater_a": round(100.0 * greater / n, 2),
        "wilcoxon_p_two_sided": None if p_value is None else round(p_value, 10),
        "bootstrap_ci95_mean_delta_low": round(ci_low, 6),
        "bootstrap_ci95_mean_delta_high": round(ci_high, 6),
        "cohen_dz": None if cohen_dz is None else round(float(cohen_dz), 6),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Saved: {path}")


def print_summary(summary_rows, component_a, component_b):
    print()
    print(f"Paired component diagnostic: {component_a} vs {component_b}")
    headers = [
        "metric",
        "n",
        "mean_delta_b_minus_a",
        "std_delta",
        "median_delta",
        "folds_b_greater_a",
        "wilcoxon_p_two_sided",
        "bootstrap_ci95_mean_delta_low",
        "bootstrap_ci95_mean_delta_high",
        "cohen_dz",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in summary_rows))
        for header in headers
    }
    print("  ".join(header.rjust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in summary_rows:
        print("  ".join(str(row[header]).rjust(widths[header]) for header in headers))


def print_cautions():
    # Bu uyarilar, post-hoc component karsilastirmasinin raporda asiri yorumlanmasini engeller.
    print()
    print("WARNING: This is paired fold-level exploratory analysis.")
    print("WARNING: n is small, usually 11 LOSO folds.")
    print("WARNING: This does not remove model-selection bias.")
    print("WARNING: Nested LOSO would be needed for unbiased component selection.")


def main():
    args = parse_args()
    results_dir = args.results_dir
    path_a = results_dir / f"csp_components_{args.component_a}_loso_results.csv"
    path_b = results_dir / f"csp_components_{args.component_b}_loso_results.csv"

    print("Results directory:", results_dir.resolve())
    print("Component A file:", path_a.resolve())
    print("Component B file:", path_b.resolve())
    print_cautions()

    try:
        rows_a, columns_a = load_component_results(path_a)
        rows_b, columns_b = load_component_results(path_b)
        paired_rows = build_paired_delta_rows(
            args.component_a,
            args.component_b,
            rows_a,
            rows_b,
        )

        auc_deltas = np.array(
            [row["delta_roc_auc_b_minus_a"] for row in paired_rows],
            dtype=float,
        )
        ba_deltas = np.array(
            [row["delta_balanced_accuracy_b_minus_a"] for row in paired_rows],
            dtype=float,
        )
        summary_rows = [
            summarize_metric("roc_auc", auc_deltas, args.n_bootstrap, args.seed),
            summarize_metric(
                "balanced_accuracy",
                ba_deltas,
                args.n_bootstrap,
                args.seed,
            ),
        ]

        for row in summary_rows:
            row["component_a"] = args.component_a
            row["component_b"] = args.component_b
            row["n_bootstrap"] = args.n_bootstrap
            row["seed"] = args.seed

        output_prefix = results_dir / args.output_prefix
        write_manifest(
            results_dir,
            script_name=Path(__file__).name,
            cli_args=vars(args),
            input_dir=results_dir,
            output_dir_value=results_dir,
            extra={
                "component_a_file": path_a,
                "component_b_file": path_b,
                "output_prefix": args.output_prefix,
                "analysis_type": "paired fold-level exploratory diagnostic",
            },
        )
        write_csv(Path(f"{output_prefix}_paired_deltas.csv"), paired_rows)
        write_csv(Path(f"{output_prefix}_summary.csv"), summary_rows)
        write_json(
            Path(f"{output_prefix}_summary.json"),
            {
                "component_a": args.component_a,
                "component_b": args.component_b,
                "results_dir": str(results_dir.resolve()),
                "component_a_columns": columns_a,
                "component_b_columns": columns_b,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "caution": (
                    "Paired fold-level exploratory analysis only; nested LOSO "
                    "would be needed for unbiased component selection."
                ),
                "summary": summary_rows,
            },
        )
        print_summary(summary_rows, args.component_a, args.component_b)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
