import argparse
import csv
from pathlib import Path
import re
import time

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from experiment_manifest import write_manifest
from train_csp_5band_baseline import (
    apply_bandpass_to_epochs,
    build_loso_folds,
    build_train_test_data,
    extract_log_variance,
    get_canonical_bands,
    sort_session_ids,
)


SCRIPT_NAME = Path(__file__).name
EXPECTED_SESSION_COUNT = 11
EXPECTED_CHANNELS = 15
EXPECTED_TIMES = 500
DEFAULT_INPUT_DIR = Path("outputs") / "window_data_wideband"
DEFAULT_OUTPUT_DIR = Path("outputs") / "xai_results"


# Bu script egitim hattini degistirmez; mevcut 5-band CSP + LDA akisinin post-hoc yorumlanabilirlik ciktilarini uretir.
# LDA katsayilari ikincil tanisal sinyal, opsiyonel permutation ise band bazli daha guvenilir post-hoc sinyaldir.
def get_default_random_seed():
    return int(getattr(config, "RANDOM_SEED", 42))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute post-hoc 5-band CSP + LDA coefficient diagnostics by band "
            "and CSP component."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing session_*_wideband_windows.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for XAI diagnostic CSV outputs.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        type=int,
        default=[4, 8],
        help="Positive CSP component settings to analyze.",
    )
    parser.add_argument(
        "--test-sessions",
        nargs="+",
        type=int,
        help="Optional LOSO test sessions to run. Omit to run all 11 folds.",
    )
    parser.add_argument(
        "--include-permutation",
        action="store_true",
        help="Also compute post-hoc band permutation importance on test folds.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=5,
        help="Permutation repeats per fold/band when --include-permutation is used.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=get_default_random_seed(),
        help="Random seed for reproducible permutation diagnostics.",
    )
    parser.add_argument(
        "--compare-ablation-dir",
        type=Path,
        help="Optional directory containing csp_components_*_loso_results.csv files.",
    )
    args = parser.parse_args()

    invalid_components = [value for value in args.components if value <= 0]
    if invalid_components:
        parser.error(f"--components values must be positive integers: {invalid_components}")

    duplicate_components = sorted(
        value for value in set(args.components)
        if args.components.count(value) > 1
    )
    if duplicate_components:
        parser.error(f"--components values must be unique: {duplicate_components}")

    too_large_components = [
        value for value in args.components
        if value > EXPECTED_CHANNELS
    ]
    if too_large_components:
        parser.error(
            f"--components values cannot exceed n_channels={EXPECTED_CHANNELS}: "
            f"{too_large_components}"
        )

    if args.test_sessions is not None:
        invalid_sessions = [
            value for value in args.test_sessions
            if value < 1 or value > EXPECTED_SESSION_COUNT
        ]
        if invalid_sessions:
            parser.error(
                "--test-sessions values must be between 1 and "
                f"{EXPECTED_SESSION_COUNT}: {invalid_sessions}"
            )
        args.test_sessions = sorted(set(args.test_sessions))

    if args.permutation_repeats <= 0:
        parser.error("--permutation-repeats must be a positive integer")

    return args


def derive_output_dir(base_output_dir, selected_test_sessions):
    if selected_test_sessions is None:
        return base_output_dir

    session_tag = "_".join(str(session) for session in selected_test_sessions)
    return base_output_dir / f"smoke_sessions_{session_tag}"


def parse_session_id(path):
    match = re.fullmatch(r"session_(\d+)_wideband_windows\.npz", path.name)
    if not match:
        raise ValueError(f"Unexpected wideband window filename: {path.name}")
    return match.group(1)


def find_session_files(input_dir):
    # XAI analizi tum 11 session'in wideband pencere dosyasini bekler.
    # Eksik veya fazla dosya, LOSO fold yapisini degistirip band karsilastirmasini bozabilir.
    files = sorted(input_dir.glob("session_*_wideband_windows.npz"))
    if len(files) != EXPECTED_SESSION_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_SESSION_COUNT} files matching "
            f"{input_dir / 'session_*_wideband_windows.npz'}, found {len(files)}"
        )

    session_files = {}
    for path in files:
        session_id = parse_session_id(path)
        if session_id in session_files:
            raise ValueError(f"Duplicate wideband window file for session {session_id}")
        session_files[session_id] = path

    expected_ids = {str(session) for session in range(1, EXPECTED_SESSION_COUNT + 1)}
    found_ids = set(session_files)
    if found_ids != expected_ids:
        raise ValueError(
            "Expected wideband window files for sessions 1..11. "
            f"Missing: {sorted(expected_ids - found_ids, key=int)}, "
            f"extra: {sorted(found_ids - expected_ids, key=int)}"
        )

    return session_files


def load_all_sessions(input_dir):
    # Burada sadece X/y dosyalari okunur ve sekil/label sozlesmesi dogrulanir.
    # CSP, scaler ve LDA fit islemleri daha sonra fold icinde train verisiyle yapilir.
    session_files = find_session_files(input_dir)
    session_data = {}

    for session_id in sort_session_ids(session_files.keys()):
        path = session_files[session_id]
        with np.load(path) as data:
            missing = [key for key in ["X", "y"] if key not in data]
            if missing:
                raise ValueError(f"{path} is missing required arrays: {missing}")
            X = np.asarray(data["X"])
            y = np.asarray(data["y"])

        if X.ndim != 3:
            raise ValueError(f"{path} X must be 3D, got shape {X.shape}")
        if X.shape[1] != EXPECTED_CHANNELS or X.shape[2] != EXPECTED_TIMES:
            raise ValueError(
                f"{path} expected X shape (*, {EXPECTED_CHANNELS}, {EXPECTED_TIMES}), "
                f"got {X.shape}"
            )
        if y.ndim != 1:
            raise ValueError(f"{path} y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"{path} X/y row mismatch: X has {X.shape[0]}, y has {y.shape[0]}"
            )

        labels = set(np.unique(y).tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(f"{path} y contains labels outside {{0, 1}}: {sorted(labels)}")

        y0 = int(np.sum(y == 0))
        y1 = int(np.sum(y == 1))
        if y0 == 0 or y1 == 0:
            raise ValueError(
                f"{path} must contain both classes. Found y0={y0}, y1={y1}"
            )

        print(f"Session {session_id}: X shape={X.shape}, y0={y0}, y1={y1}")

        session_data[session_id] = {
            "X": X,
            "y": y,
        }

    return session_data


def build_feature_mapping(csp_components):
    # Feature mapping, concatenated CSP feature kolonlarini band ve component etiketlerine geri baglar.
    # Bu eslesme XAI CSV'lerinin hangi kolonun hangi banttan geldigini aciklamasi icin gereklidir.
    mapping = []
    bands = get_canonical_bands()
    for band_index, (band, _, _) in enumerate(bands):
        for component_index in range(csp_components):
            global_feature_index = band_index * csp_components + component_index
            mapping.append(
                {
                    "band": band,
                    "band_index": band_index,
                    "csp_component": component_index,
                    "global_feature_index": global_feature_index,
                }
            )
    return mapping


def build_feature_mapping_records(component_settings, provenance):
    records = []
    for component_setting in component_settings:
        for feature in build_feature_mapping(component_setting):
            records.append(
                {
                    **provenance,
                    "component_setting": int(component_setting),
                    **feature,
                }
            )
    return records


def fit_transform_5band_features(X_train, y_train, X_test, csp_components):
    # Bu fonksiyon training pipeline'ini XAI icin tekrarlar.
    # Her bandin CSP modeli yalnizca train veride fit edilir, test verisi sadece transform edilir.
    bands = get_canonical_bands()
    train_blocks = []
    test_blocks = []

    for band, low_freq, high_freq in bands:
        print(f"    Band {band}: {low_freq}-{high_freq} Hz")
        X_train_band = apply_bandpass_to_epochs(
            X_train,
            sfreq=float(config.TARGET_SFREQ),
            low_freq=low_freq,
            high_freq=high_freq,
        )
        X_test_band = apply_bandpass_to_epochs(
            X_test,
            sfreq=float(config.TARGET_SFREQ),
            low_freq=low_freq,
            high_freq=high_freq,
        )

        csp_model = CSP(
            n_components=csp_components,
            reg="ledoit_wolf",
            log=None,
            transform_into="csp_space",
            norm_trace=False,
        )
        # CSP sinif etiketlerini kullandigi icin test session fit'e katilmaz.
        # Bu kural XAI tekrar kosusunda da ana baseline ile ayni sekilde korunur.
        X_train_csp = csp_model.fit_transform(X_train_band, y_train)
        X_test_csp = csp_model.transform(X_test_band)

        train_blocks.append(extract_log_variance(X_train_csp))
        test_blocks.append(extract_log_variance(X_test_csp))

    return np.concatenate(train_blocks, axis=1), np.concatenate(test_blocks, axis=1)


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def score_predictions(y_true, y_pred, y_score):
    auc = roc_auc_score(y_true, y_score)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return auc, bal_acc, cm


def make_provenance(args, output_dir, selected_test_sessions):
    return {
        "script_name": SCRIPT_NAME,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "n_sessions_loaded": EXPECTED_SESSION_COUNT,
        "selected_test_sessions": ",".join(str(s) for s in selected_test_sessions),
        "random_seed": int(args.random_seed),
        "include_permutation": bool(args.include_permutation),
    }


def record_common_metrics(
    component_setting,
    fold_id,
    fold,
    y_train,
    y_test,
    auc,
    bal_acc,
    cm,
):
    return {
        "component_setting": int(component_setting),
        "fold_id": int(fold_id),
        "test_session": fold["test_session"],
        "roc_auc": round(float(auc), 6),
        "balanced_accuracy": round(float(bal_acc), 6),
        "cm_00": int(cm[0, 0]),
        "cm_01": int(cm[0, 1]),
        "cm_10": int(cm[1, 0]),
        "cm_11": int(cm[1, 1]),
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "train_sessions": ",".join(fold["train_sessions"]),
    }


def build_feature_records(component_setting, fold_id, fold, coef, metrics, provenance):
    # LDA katsayilari feature bazinda kaydedilir; mutlak deger yalnizca goreli onem tanisidir.
    # Bu kayit causal yorum degil, hangi band/component kolonlarinin modele agirlik verdigini gosterir.
    records = []
    feature_mapping = build_feature_mapping(component_setting)
    for feature in feature_mapping:
        global_feature_index = feature["global_feature_index"]
        lda_coef = float(coef[global_feature_index])
        records.append(
            {
                **provenance,
                **metrics,
                "diagnostic_type": "lda_abs_coef_secondary",
                "band": feature["band"],
                "band_index": feature["band_index"],
                "csp_component": feature["csp_component"],
                "global_feature_index": global_feature_index,
                "lda_coef": round(lda_coef, 10),
                "abs_lda_coef": round(abs(lda_coef), 10),
            }
        )
    return records


def build_band_records(component_setting, feature_records, metrics, provenance):
    records = []
    bands = get_canonical_bands()
    total_abs = sum(row["abs_lda_coef"] for row in feature_records)

    for band, _, _ in bands:
        band_values = [
            row["abs_lda_coef"] for row in feature_records
            if row["band"] == band
        ]
        sum_abs = float(np.sum(band_values))
        mean_abs = float(np.mean(band_values))
        normalized = 0.0 if total_abs == 0 else sum_abs / total_abs
        records.append(
            {
                **provenance,
                **metrics,
                "diagnostic_type": "lda_abs_coef_secondary",
                "band": band,
                "mean_abs_coef": round(mean_abs, 10),
                "sum_abs_coef": round(sum_abs, 10),
                "normalized_sum_abs_coef": round(float(normalized), 10),
            }
        )
    return records


def summarize_band_importance(band_records, provenance):
    summary = []
    keys = sorted(
        {(row["component_setting"], row["band"]) for row in band_records},
        key=lambda item: (item[0], band_order(item[1])),
    )

    for component_setting, band in keys:
        rows = [
            row for row in band_records
            if row["component_setting"] == component_setting and row["band"] == band
        ]
        normalized = np.array([row["normalized_sum_abs_coef"] for row in rows], dtype=float)
        sum_abs = np.array([row["sum_abs_coef"] for row in rows], dtype=float)
        mean_abs = np.array([row["mean_abs_coef"] for row in rows], dtype=float)
        summary.append(
            {
                **provenance,
                "diagnostic_type": "lda_abs_coef_secondary",
                "component_setting": int(component_setting),
                "band": band,
                "mean_normalized_sum_abs_coef": round(float(np.mean(normalized)), 10),
                "std_normalized_sum_abs_coef": round(float(np.std(normalized)), 10),
                "mean_sum_abs_coef": round(float(np.mean(sum_abs)), 10),
                "std_sum_abs_coef": round(float(np.std(sum_abs)), 10),
                "mean_abs_coef": round(float(np.mean(mean_abs)), 10),
                "std_abs_coef": round(float(np.std(mean_abs)), 10),
                "fold_count": len(rows),
            }
        )
    return summary


def summarize_component_importance(feature_records, provenance):
    summary = []
    keys = sorted(
        {
            (row["component_setting"], row["band"], row["csp_component"])
            for row in feature_records
        },
        key=lambda item: (item[0], band_order(item[1]), item[2]),
    )

    for component_setting, band, csp_component in keys:
        rows = [
            row for row in feature_records
            if row["component_setting"] == component_setting
            and row["band"] == band
            and row["csp_component"] == csp_component
        ]
        values = np.array([row["abs_lda_coef"] for row in rows], dtype=float)
        summary.append(
            {
                **provenance,
                "diagnostic_type": "lda_abs_coef_secondary",
                "component_setting": int(component_setting),
                "band": band,
                "csp_component": int(csp_component),
                "mean_abs_coef": round(float(np.mean(values)), 10),
                "std_abs_coef": round(float(np.std(values)), 10),
                "fold_count": len(rows),
            }
        )
    return summary


def band_order(band_name):
    names = [band for band, _, _ in get_canonical_bands()]
    return names.index(band_name)


def build_fold_metric_record(metrics, provenance):
    return {
        **provenance,
        **metrics,
    }


def run_band_permutation_importance(
    args,
    component_setting,
    fold_id,
    fold,
    y_test,
    X_test_scaled,
    lda_model,
    baseline_auc,
    baseline_bal_acc,
    metrics,
    provenance,
    rng,
):
    # Permutation importance test fold'unda tek bandin feature kolonlarini karistirir.
    # Skor dususu, egitilmis modelin o band bilgisine ne kadar dayandigini post-hoc olarak ozetler.
    records = []
    bands = get_canonical_bands()
    repeats = int(args.permutation_repeats)

    for band_index, (band, _, _) in enumerate(bands):
        start = band_index * component_setting
        stop = start + component_setting
        auc_values = []
        bal_acc_values = []

        for _ in range(repeats):
            X_permuted = X_test_scaled.copy()
            shuffled_rows = rng.permutation(X_permuted.shape[0])
            X_permuted[:, start:stop] = X_permuted[shuffled_rows, start:stop]
            y_pred = lda_model.predict(X_permuted)
            y_score = predict_scores(lda_model, X_permuted)
            auc, bal_acc, _ = score_predictions(y_test, y_pred, y_score)
            auc_values.append(float(auc))
            bal_acc_values.append(float(bal_acc))

        mean_auc = float(np.mean(auc_values))
        mean_bal_acc = float(np.mean(bal_acc_values))
        records.append(
            {
                **provenance,
                **metrics,
                "band": band,
                "permutation_repeats": repeats,
                "baseline_roc_auc": round(float(baseline_auc), 6),
                "baseline_balanced_accuracy": round(float(baseline_bal_acc), 6),
                "mean_permuted_roc_auc": round(mean_auc, 6),
                "mean_permuted_balanced_accuracy": round(mean_bal_acc, 6),
                "mean_auc_drop": round(float(baseline_auc - mean_auc), 6),
                "mean_balanced_accuracy_drop": round(float(baseline_bal_acc - mean_bal_acc), 6),
            }
        )

    return records


def summarize_permutation_importance(permutation_records, provenance):
    summary = []
    keys = sorted(
        {(row["component_setting"], row["band"]) for row in permutation_records},
        key=lambda item: (item[0], band_order(item[1])),
    )

    for component_setting, band in keys:
        rows = [
            row for row in permutation_records
            if row["component_setting"] == component_setting and row["band"] == band
        ]
        auc_drops = np.array([row["mean_auc_drop"] for row in rows], dtype=float)
        ba_drops = np.array(
            [row["mean_balanced_accuracy_drop"] for row in rows],
            dtype=float,
        )
        summary.append(
            {
                **provenance,
                "diagnostic_type": "band_permutation_primary",
                "component_setting": int(component_setting),
                "band": band,
                "mean_auc_drop": round(float(np.mean(auc_drops)), 6),
                "std_auc_drop": round(float(np.std(auc_drops)), 6),
                "mean_balanced_accuracy_drop": round(float(np.mean(ba_drops)), 6),
                "std_balanced_accuracy_drop": round(float(np.std(ba_drops)), 6),
                "fold_count": len(rows),
            }
        )
    return summary


def write_csv(path, rows):
    if not rows:
        print(f"Skipping empty output: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def read_csv_rows(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def compare_ablation_metrics(compare_dir, fold_metric_records, output_dir, provenance):
    required_columns = {
        "test_session",
        "roc_auc",
        "balanced_accuracy",
        "cm_00",
        "cm_01",
        "cm_10",
        "cm_11",
        "n_train_samples",
        "n_test_samples",
    }
    compare_fields = [
        "roc_auc",
        "balanced_accuracy",
        "cm_00",
        "cm_01",
        "cm_10",
        "cm_11",
        "n_train_samples",
        "n_test_samples",
    ]
    ablation_cache = {}
    comparison_rows = []

    for xai_row in fold_metric_records:
        component_setting = int(xai_row["component_setting"])
        test_session = str(xai_row["test_session"])
        if component_setting not in ablation_cache:
            path = compare_dir / f"csp_components_{component_setting}_loso_results.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing ablation comparison file: {path}")
            rows, fieldnames = read_csv_rows(path)
            missing = required_columns - set(fieldnames)
            if missing:
                raise ValueError(
                    f"{path} is missing required columns: {sorted(missing)}"
                )
            ablation_cache[component_setting] = {
                row["test_session"]: row for row in rows
            }

        ablation_rows = ablation_cache[component_setting]
        if test_session not in ablation_rows:
            raise ValueError(
                f"No ablation row for component={component_setting}, "
                f"test_session={test_session}"
            )

        ablation_row = ablation_rows[test_session]
        comparison = {
            **provenance,
            "component_setting": component_setting,
            "fold_id": xai_row["fold_id"],
            "test_session": test_session,
        }

        for field in compare_fields:
            xai_value = float(xai_row[field])
            ablation_value = float(ablation_row[field])
            comparison[f"xai_{field}"] = xai_row[field]
            comparison[f"ablation_{field}"] = ablation_row[field]
            comparison[f"abs_diff_{field}"] = round(abs(xai_value - ablation_value), 6)

        cm_mismatch = any(
            comparison[f"abs_diff_{field}"] != 0
            for field in ["cm_00", "cm_01", "cm_10", "cm_11"]
        )
        sample_mismatch = any(
            comparison[f"abs_diff_{field}"] != 0
            for field in ["n_train_samples", "n_test_samples"]
        )
        comparison["warning"] = ""
        if cm_mismatch or sample_mismatch:
            comparison["warning"] = "confusion_matrix_or_sample_count_mismatch"
            print(
                "WARNING: Ablation comparison mismatch for "
                f"component={component_setting}, test_session={test_session}"
            )

        comparison_rows.append(comparison)

    write_csv(output_dir / "csp_xai_ablation_metric_check.csv", comparison_rows)
    print_ablation_comparison_table(comparison_rows)
    return comparison_rows


def print_ablation_comparison_table(rows):
    print()
    print("Ablation metric comparison:")
    headers = [
        "component_setting",
        "test_session",
        "abs_diff_roc_auc",
        "abs_diff_balanced_accuracy",
        "abs_diff_cm_00",
        "abs_diff_cm_01",
        "abs_diff_cm_10",
        "abs_diff_cm_11",
    ]
    print_compact_table(rows, headers)


def print_compact_table(rows, headers):
    if not rows:
        return

    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print("  ".join(header.rjust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).rjust(widths[header]) for header in headers))


def check_normalized_band_sums(band_records):
    grouped = {}
    for row in band_records:
        key = (row["component_setting"], row["fold_id"], row["test_session"])
        grouped.setdefault(key, 0.0)
        grouped[key] += float(row["normalized_sum_abs_coef"])

    for key, total in sorted(grouped.items()):
        if not np.isclose(total, 1.0, atol=1e-6):
            print(
                "WARNING: normalized_sum_abs_coef does not sum to 1 for "
                f"component={key[0]}, fold={key[1]}, test_session={key[2]}: {total}"
            )


def warn_extreme_coefficient_dominance(feature_records):
    grouped = {}
    for row in feature_records:
        key = (row["component_setting"], row["fold_id"], row["test_session"])
        grouped.setdefault(key, []).append(float(row["abs_lda_coef"]))

    for key, values in sorted(grouped.items()):
        abs_values = np.array(values, dtype=float)
        median_abs = float(np.median(abs_values))
        max_abs = float(np.max(abs_values))
        unstable = (median_abs == 0.0 and max_abs > 0.0) or (
            median_abs > 0.0 and max_abs / median_abs > 1e6
        )
        if unstable:
            print(
                "WARNING: Coefficient magnitudes may be numerically unstable. "
                f"component={key[0]}, fold={key[1]}, test_session={key[2]}, "
                f"max_abs={max_abs:.6g}, median_abs={median_abs:.6g}"
            )


def print_summary_table(summary_rows):
    print()
    print("Final band summary by component setting:")
    headers = [
        "component_setting",
        "band",
        "mean_normalized_sum_abs_coef",
        "mean_sum_abs_coef",
        "mean_abs_coef",
        "fold_count",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in summary_rows))
        for header in headers
    }
    print("  ".join(header.rjust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in summary_rows:
        print("  ".join(str(row[header]).rjust(widths[header]) for header in headers))


def print_permutation_summary_table(summary_rows):
    print()
    print("Permutation importance summary by component setting:")
    headers = [
        "component_setting",
        "band",
        "mean_auc_drop",
        "mean_balanced_accuracy_drop",
        "fold_count",
    ]
    print_compact_table(summary_rows, headers)


def main():
    start_time = time.time()
    args = parse_args()
    selected_test_sessions = (
        args.test_sessions
        if args.test_sessions is not None
        else list(range(1, EXPECTED_SESSION_COUNT + 1))
    )
    output_dir = derive_output_dir(args.output_dir, args.test_sessions)
    provenance = make_provenance(args, output_dir, selected_test_sessions)
    rng = np.random.default_rng(args.random_seed)
    write_manifest(
        output_dir,
        script_name=SCRIPT_NAME,
        cli_args=vars(args),
        input_dir=args.input_dir,
        output_dir_value=output_dir,
        extra={
            "selected_test_sessions": selected_test_sessions,
            "component_settings": args.components,
        },
    )

    print("Current working directory:", Path.cwd())
    print("Input directory:", args.input_dir.resolve())
    print("Output directory:", output_dir.resolve())
    print(
        "WARNING: Interpretation assumes check_project_outputs.py has passed "
        "for this outputs/ directory."
    )
    print(
        "WARNING: LDA coefficient magnitude is a secondary post-hoc diagnostic "
        "and may be numerically unstable; use permutation importance as the "
        "primary band-level XAI signal when available."
    )
    if args.include_permutation:
        print(
            "Permutation importance is a post-hoc diagnostic only, not model "
            "selection or parameter tuning."
        )

    session_data = load_all_sessions(args.input_dir)
    loaded_sessions = sort_session_ids(list(session_data.keys()))
    print("Loaded sessions:", loaded_sessions)
    print("Selected test sessions:", selected_test_sessions)
    print("Component settings:", args.components)
    print("Include permutation:", args.include_permutation)

    folds = build_loso_folds(session_data)
    selected_session_ids = {str(session) for session in selected_test_sessions}
    selected_folds = [
        (fold_id, fold) for fold_id, fold in enumerate(folds, start=1)
        if fold["test_session"] in selected_session_ids
    ]
    if not selected_folds:
        raise ValueError(f"No LOSO folds selected by --test-sessions={selected_test_sessions}")

    feature_records = []
    band_records = []
    fold_metric_records = []
    permutation_records = []

    for component_setting in args.components:
        print(f"\n===== CSP components: {component_setting} =====")
        if component_setting > EXPECTED_CHANNELS:
            raise ValueError(
                f"csp_components ({component_setting}) cannot exceed n_channels "
                f"({EXPECTED_CHANNELS})"
            )

        for fold_id, fold in selected_folds:
            print(
                f"\nFold {fold_id}: train={fold['train_sessions']} "
                f"test={fold['test_session']}"
            )
            X_train, y_train, X_test, y_test = build_train_test_data(session_data, fold)
            print("  X_train shape:", X_train.shape)
            print("  X_test shape:", X_test.shape)

            X_train_feat, X_test_feat = fit_transform_5band_features(
                X_train,
                y_train,
                X_test,
                csp_components=component_setting,
            )
            expected_features = len(get_canonical_bands()) * component_setting
            if X_train_feat.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {X_train_feat.shape[1]}"
                )

            # XAI tekrar kosusunda da scaler yalnizca train feature'larinda fit edilir.
            # Test fold'unun olcek bilgisi katsayi veya metrik hesabina sizdirilmez.
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_feat)
            X_test_scaled = scaler.transform(X_test_feat)

            lda_model = LinearDiscriminantAnalysis()
            lda_model.fit(X_train_scaled, y_train)
            y_pred = lda_model.predict(X_test_scaled)
            y_score = predict_scores(lda_model, X_test_scaled)
            auc, bal_acc, cm = score_predictions(y_test, y_pred, y_score)
            print(f"  ROC-AUC: {auc:.6f}")
            print(f"  Balanced Accuracy: {bal_acc:.6f}")

            coef = np.asarray(lda_model.coef_).reshape(-1)
            if coef.shape[0] != expected_features:
                raise ValueError(
                    f"LDA coefficient count mismatch: expected {expected_features}, "
                    f"got {coef.shape[0]}"
                )

            metrics = record_common_metrics(
                component_setting,
                fold_id,
                fold,
                y_train,
                y_test,
                auc,
                bal_acc,
                cm,
            )
            fold_metric_records.append(build_fold_metric_record(metrics, provenance))

            fold_feature_records = build_feature_records(
                component_setting,
                fold_id,
                fold,
                coef,
                metrics,
                provenance,
            )
            feature_records.extend(fold_feature_records)
            band_records.extend(
                build_band_records(
                    component_setting,
                    fold_feature_records,
                    metrics,
                    provenance,
                )
            )

            if args.include_permutation:
                permutation_records.extend(
                    run_band_permutation_importance(
                        args,
                        component_setting,
                        fold_id,
                        fold,
                        y_test,
                        X_test_scaled,
                        lda_model,
                        auc,
                        bal_acc,
                        metrics,
                        provenance,
                        rng,
                    )
                )

    band_summary_records = summarize_band_importance(band_records, provenance)
    component_summary_records = summarize_component_importance(feature_records, provenance)
    feature_mapping_records = build_feature_mapping_records(args.components, provenance)
    permutation_summary_records = (
        summarize_permutation_importance(permutation_records, provenance)
        if args.include_permutation
        else []
    )

    check_normalized_band_sums(band_records)
    warn_extreme_coefficient_dominance(feature_records)

    # Asagidaki CSV'ler tez raporu ve hata kontrolu icin ayrik artefaktlar olarak saklanir.
    # Fold-level kayitlar, ozet tablolari tekrar uretmeyi mumkun kilar.
    write_csv(output_dir / "csp_feature_mapping.csv", feature_mapping_records)
    write_csv(output_dir / "csp_feature_importance_fold_level.csv", feature_records)
    write_csv(output_dir / "csp_band_importance_fold_level.csv", band_records)
    write_csv(output_dir / "csp_band_importance_summary.csv", band_summary_records)
    write_csv(output_dir / "csp_component_importance_summary.csv", component_summary_records)
    write_csv(output_dir / "csp_xai_fold_metrics.csv", fold_metric_records)
    if args.include_permutation:
        write_csv(output_dir / "csp_band_permutation_importance.csv", permutation_records)
        write_csv(output_dir / "csp_band_permutation_importance_summary.csv", permutation_summary_records)

    if args.compare_ablation_dir is not None:
        compare_ablation_metrics(
            args.compare_ablation_dir,
            fold_metric_records,
            output_dir,
            provenance,
        )

    print_summary_table(band_summary_records)
    if args.include_permutation:
        print_permutation_summary_table(permutation_summary_records)
    print()
    print(
        "Note: LDA coefficient magnitude is a secondary relative post-hoc "
        "diagnostic, not causal proof."
    )
    if args.include_permutation:
        print(
            "Note: Permutation importance is the primary band-level XAI "
            "diagnostic in this script, but it is still post-hoc and not "
            "model selection."
        )
    print(f"Runtime seconds: {time.time() - start_time:.2f}")


if __name__ == "__main__":
    main()
