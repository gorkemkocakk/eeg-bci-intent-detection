import csv
from collections import Counter
from pathlib import Path
import sys

import numpy as np


# =========================================================
# 1) BEKLENEN CIKTI SOZLESMESI
# =========================================================
# Bu script model egitmez; uretilen CSV/NPZ artefaktlarinin proje metodolojisine uydugunu denetler.
# Amac, eski hatali ciktinin veya eksik session dosyasinin rapora sessizce karismasini engellemektir.
EXPECTED_SESSIONS = range(1, 12)
EXPECTED_COUNT = 11
OUTPUT_DIR = Path("outputs")
TRIAL_DIR = OUTPUT_DIR / "trial_tables"
LABEL_DIR = OUTPUT_DIR / "label_tables"
WINDOW_DIR = OUTPUT_DIR / "window_data"
WIDEBAND_WINDOW_DIR = OUTPUT_DIR / "window_data_wideband"
TRIAL_REQUIRED_COLUMNS = {
    "triallength_sec",
    "feedback_duration_sec",
    "feedback_semantic",
    "feedback_start_sec",
    "feedback_end_sec",
}
EXPECTED_FEEDBACK_SEMANTIC = "feedback_start_plus_triallength"
DURATION_TOLERANCE = 1e-6
METADATA_COMPARE_COLUMNS = [
    "trial_id",
    "session_id",
    "window_start",
    "window_end",
]


def fail(message, category, failed_categories):
    # Hatalar kategori bazinda toplanir; boylece final verdict hangi kontrol grubunun bozuldugunu gosterir.
    print(f"ERROR: {message}")
    failed_categories.add(category)
    return False


def warn(message):
    print(f"WARNING: {message}")


def count_labels(y):
    return int(y.size), int(np.sum(y == 0)), int(np.sum(y == 1))


def load_y(path):
    with np.load(path) as data:
        if "y" not in data:
            raise KeyError(f"{path} does not contain a 'y' array")
        return np.asarray(data["y"])


def read_csv_rows(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def check_file_count(folder, pattern, expected_count, category, failed_categories):
    files = sorted(folder.glob(pattern)) if folder.is_dir() else []
    ok = len(files) == expected_count
    if not ok:
        fail(
            f"Expected {expected_count} files matching {folder / pattern}, found {len(files)}",
            category,
            failed_categories,
        )
    return ok


def print_paths():
    print("Current working directory:", Path.cwd().resolve())
    print("Outputs directory:", OUTPUT_DIR.resolve())
    print("Trial tables directory:", TRIAL_DIR.resolve())
    print("Label tables directory:", LABEL_DIR.resolve())
    print("Window data directory:", WINDOW_DIR.resolve())
    print("Wideband window data directory:", WIDEBAND_WINDOW_DIR.resolve())


def print_label_table(rows):
    headers = [
        "session",
        "normal_n",
        "normal_l0",
        "normal_l1",
        "wide_n",
        "wide_l0",
        "wide_l1",
        "match",
    ]
    print_table(rows, headers)


def print_table(rows, headers):
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }

    header_line = "  ".join(header.rjust(widths[header]) for header in headers)
    separator = "  ".join("-" * widths[header] for header in headers)
    print(header_line)
    print(separator)

    for row in rows:
        print("  ".join(str(row[header]).rjust(widths[header]) for header in headers))


def check_buggy_restored_folders():
    buggy_folders = sorted(
        path for path in Path.cwd().glob("outputs_buggy_restored_*") if path.is_dir()
    )
    for folder in buggy_folders:
        warn(f"Found restored buggy output folder: {folder}")


def check_required_output_files(failed_categories):
    # Her ana asama icin 11 session dosyasi beklenir.
    # Dosya sayisi eksikse LOSO fold sayisi ve raporlanan ortalamalar degisir.
    ok = True
    ok = check_file_count(
        TRIAL_DIR,
        "session_*_trial_table.csv",
        EXPECTED_COUNT,
        "missing required files",
        failed_categories,
    ) and ok
    ok = check_file_count(
        LABEL_DIR,
        "session_*_labels.csv",
        EXPECTED_COUNT,
        "missing required files",
        failed_categories,
    ) and ok
    ok = check_file_count(
        WINDOW_DIR,
        "session_*_windows.npz",
        EXPECTED_COUNT,
        "missing required files",
        failed_categories,
    ) and ok
    ok = check_file_count(
        WIDEBAND_WINDOW_DIR,
        "session_*_wideband_windows.npz",
        EXPECTED_COUNT,
        "missing required files",
        failed_categories,
    ) and ok
    return ok


def check_window_labels(failed_categories):
    # Normal ve wideband pencere dosyalari ayni segmentlerden turemis olmali.
    # Label sayilari ayrisirsa 5-band deneyleri bandpower/CSP baseline ile adil karsilastirilamaz.
    rows = []
    ok = True

    for session in EXPECTED_SESSIONS:
        label_path = LABEL_DIR / f"session_{session}_labels.csv"
        normal_path = WINDOW_DIR / f"session_{session}_windows.npz"
        wide_path = WIDEBAND_WINDOW_DIR / f"session_{session}_wideband_windows.npz"

        for path in [label_path, normal_path, wide_path]:
            if not path.is_file():
                ok = fail(
                    f"Missing required output file: {path}",
                    "missing required files",
                    failed_categories,
                ) and ok

        if not normal_path.is_file() or not wide_path.is_file():
            continue

        try:
            normal_y = load_y(normal_path)
            wide_y = load_y(wide_path)
        except Exception as exc:
            ok = fail(str(exc), "window y labels", failed_categories) and ok
            continue

        normal_n, normal_l0, normal_l1 = count_labels(normal_y)
        wide_n, wide_l0, wide_l1 = count_labels(wide_y)
        match = (
            normal_n == wide_n
            and normal_l0 == wide_l0
            and normal_l1 == wide_l1
        )
        if not match:
            ok = fail(
                f"Session {session} normal/wideband y counts do not match",
                "window y labels",
                failed_categories,
            ) and ok

        rows.append(
            {
                "session": session,
                "normal_n": normal_n,
                "normal_l0": normal_l0,
                "normal_l1": normal_l1,
                "wide_n": wide_n,
                "wide_l0": wide_l0,
                "wide_l1": wide_l1,
                "match": match,
            }
        )

    if rows:
        print()
        print("Window label count comparison:")
        print_label_table(rows)

    return ok


def check_trial_tables(failed_categories):
    # Trial tablosu, feedback baslangici ve triallength mantiginin dogru parse edildigini denetler.
    # Buradaki semantic kontrol, feedback suresinin yanlis yorumlandigi eski bug'lari yakalamak icindir.
    ok = True
    rows = []

    for session in EXPECTED_SESSIONS:
        path = TRIAL_DIR / f"session_{session}_trial_table.csv"
        if not path.is_file():
            ok = fail(
                f"Missing required trial table: {path}",
                "missing required files",
                failed_categories,
            ) and ok
            continue

        try:
            trial_rows, fieldnames = read_csv_rows(path)
        except Exception as exc:
            ok = fail(str(exc), "trial table semantics", failed_categories) and ok
            continue

        missing_columns = sorted(TRIAL_REQUIRED_COLUMNS - set(fieldnames))
        if missing_columns:
            ok = fail(
                f"{path} is missing required columns: {', '.join(missing_columns)}",
                "trial table semantics",
                failed_categories,
            ) and ok
            continue

        semantic_bad = 0
        duration_mismatch = 0
        nonpositive_duration = 0
        duration_gt_10 = 0
        duration_gt_20 = 0

        for row_number, row in enumerate(trial_rows, start=2):
            if row["feedback_semantic"] != EXPECTED_FEEDBACK_SEMANTIC:
                semantic_bad += 1

            try:
                feedback_duration = float(row["feedback_duration_sec"])
                trial_length = float(row["triallength_sec"])
            except ValueError:
                ok = fail(
                    f"{path} has non-numeric duration on CSV row {row_number}",
                    "trial table semantics",
                    failed_categories,
                ) and ok
                continue

            if abs(feedback_duration - trial_length) > DURATION_TOLERANCE:
                duration_mismatch += 1
            if feedback_duration <= 0:
                nonpositive_duration += 1
            if feedback_duration > 10:
                duration_gt_10 += 1
            if feedback_duration > 20:
                duration_gt_20 += 1

        if semantic_bad:
            ok = fail(
                f"{path} has {semantic_bad} rows with unexpected feedback_semantic",
                "trial table semantics",
                failed_categories,
            ) and ok
        if duration_mismatch:
            ok = fail(
                f"{path} has {duration_mismatch} feedback durations not matching triallength_sec",
                "trial table semantics",
                failed_categories,
            ) and ok
        if nonpositive_duration:
            ok = fail(
                f"{path} has {nonpositive_duration} non-positive feedback durations",
                "trial table semantics",
                failed_categories,
            ) and ok
        if duration_gt_10:
            warn(f"{path} has {duration_gt_10} feedback durations > 10 seconds")
        if duration_gt_20:
            ok = fail(
                f"{path} has {duration_gt_20} feedback durations > 20 seconds; this resembles the old bug",
                "trial table semantics",
                failed_categories,
            ) and ok

        rows.append(
            {
                "session": session,
                "rows": len(trial_rows),
                "bad_semantic": semantic_bad,
                "duration_mismatch": duration_mismatch,
                "nonpositive": nonpositive_duration,
                "gt_10": duration_gt_10,
                "gt_20": duration_gt_20,
            }
        )

    if rows:
        print()
        print("Trial table semantic summary:")
        print_table(
            rows,
            [
                "session",
                "rows",
                "bad_semantic",
                "duration_mismatch",
                "nonpositive",
                "gt_10",
                "gt_20",
            ],
        )

    return ok


def value_counts(rows, column):
    return Counter(row[column] for row in rows)


def compare_optional_counts(column, normal_rows, normal_fields, wide_rows, wide_fields, session):
    normal_has = column in normal_fields
    wide_has = column in wide_fields

    if not normal_has and not wide_has:
        return True, None
    if normal_has != wide_has:
        return False, f"Session {session} metadata column presence differs for {column}"
    if value_counts(normal_rows, column) != value_counts(wide_rows, column):
        return False, f"Session {session} metadata {column} counts differ"
    return True, None


def check_window_metadata(failed_categories):
    # Metadata eslesmesi, X/y satirlarinin hangi trial ve zaman araligindan geldigini korur.
    # Normal ve wideband metadata ayrisirsa sonraki model tahminleri yanlis pencereye baglanabilir.
    ok = True

    for session in EXPECTED_SESSIONS:
        normal_path = WINDOW_DIR / f"session_{session}_window_metadata.csv"
        wide_path = WIDEBAND_WINDOW_DIR / f"session_{session}_wideband_window_metadata.csv"

        if not normal_path.is_file() or not wide_path.is_file():
            warn(
                "Missing metadata for session "
                f"{session}: normal_exists={normal_path.is_file()}, "
                f"wideband_exists={wide_path.is_file()}"
            )
            continue

        try:
            normal_rows, normal_fields = read_csv_rows(normal_path)
            wide_rows, wide_fields = read_csv_rows(wide_path)
        except Exception as exc:
            ok = fail(str(exc), "window metadata", failed_categories) and ok
            continue

        if len(normal_rows) != len(wide_rows):
            ok = fail(
                f"Session {session} metadata row counts differ: "
                f"normal={len(normal_rows)}, wideband={len(wide_rows)}",
                "window metadata",
                failed_categories,
            ) and ok

        for column in ["label", "segment_type"]:
            counts_ok, message = compare_optional_counts(
                column,
                normal_rows,
                normal_fields,
                wide_rows,
                wide_fields,
                session,
            )
            if not counts_ok:
                ok = fail(message, "window metadata", failed_categories) and ok

        for column in METADATA_COMPARE_COLUMNS:
            if column not in normal_fields or column not in wide_fields:
                continue

            normal_values = [row[column] for row in normal_rows]
            wide_values = [row[column] for row in wide_rows]
            if normal_values != wide_values:
                ok = fail(
                    f"Session {session} metadata values differ for shared column {column}",
                    "window metadata",
                    failed_categories,
                ) and ok

    return ok


def print_verdict(failed_categories):
    print()
    if failed_categories:
        print("FAIL: clean outputs are not internally consistent")
        print("Failed categories:")
        for category in sorted(failed_categories):
            print(f"- {category}")
        return False

    print("PASS: clean outputs look internally consistent")
    return True


def main():
    # Kontroller sadece okuma ve raporlama yapar; cikti dosyalarini degistirmez.
    # Hata varsa non-zero exit ile batch/CI benzeri akislarin durmasini saglar.
    failed_categories = set()

    print_paths()
    check_buggy_restored_folders()

    check_required_output_files(failed_categories)
    check_window_labels(failed_categories)
    check_trial_tables(failed_categories)
    check_window_metadata(failed_categories)

    if not print_verdict(failed_categories):
        sys.exit(1)


if __name__ == "__main__":
    main()
