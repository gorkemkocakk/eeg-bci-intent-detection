import json
import os
import re
import subprocess
import sys

import config
from load_stieger import load_stieger_subject, sort_names


# =========================================================
# 1) SABIT AKIS
# =========================================================

PIPELINE_SCRIPTS = [
    "parse_trials.py",
    "build_labels.py",
    "windowing.py",
    "features_bandpower.py",
]
# Bu sira dosya bagimliliklarini temsil eder:
# trial table -> label table -> EEG windows -> bandpower features.


# =========================================================
# 2) BASIT CLI
# =========================================================

def normalize_session_name(raw_name):
    """
    Session adini mevcut CLI pattern'ine gore normalize eder.

    Mevcut scriptler pratikte "1", "2" gibi session id bekliyor.
    Bu yuzden "session_1" gibi adlar gelirse "1"e ceviriyoruz.
    """
    text = str(raw_name).strip()
    match = re.search(r"(\d+)$", text)

    if match:
        return str(int(match.group(1)))

    return text


def parse_cli_args():
    """
    Basit CLI argumanlarini parse eder.

    Kullanim:
    python run_batch_pipeline.py
    python run_batch_pipeline.py 1
    python run_batch_pipeline.py 1 3 11

    Not:
    - Arguman verilmezse config.SUBJECT_ID ve tum session'lar kullanilir.
    - Aralik islenmek istenirse: <subject_id> <start_session> <end_session>
    """
    if len(sys.argv) == 1:
        return int(config.SUBJECT_ID), None, None

    if len(sys.argv) == 2:
        return int(sys.argv[1]), None, None

    if len(sys.argv) == 4:
        subject_id = int(sys.argv[1])
        start_session = int(normalize_session_name(sys.argv[2]))
        end_session = int(normalize_session_name(sys.argv[3]))

        if start_session > end_session:
            raise ValueError("Session araliginda baslangic, bitisten buyuk olamaz.")

        return subject_id, start_session, end_session

    raise ValueError(
        "Kullanim: python run_batch_pipeline.py [subject_id] [start_session end_session]"
    )


# =========================================================
# 3) SESSION LISTESI
# =========================================================

def get_sessions_from_dataset(subject_id):
    """
    Session listesini en dogal kaynaktan, yani dataset'ten alir.
    """
    subject_data = load_stieger_subject(subject_id)
    session_names = sort_names(list(subject_data.keys()))

    # Bellek temizligi
    del subject_data

    normalized = [normalize_session_name(x) for x in session_names]
    normalized = sort_names(list(set(normalized)))
    return [str(x) for x in normalized]


def get_sessions_from_inventory(subject_id):
    """
    Dataset okuma basarisiz olursa inventory json'dan session listesi alir.
    """
    inventory_path = os.path.join(
        config.OUTPUT_DIR,
        f"subject_{subject_id}_inventory.json"
    )

    if not os.path.exists(inventory_path):
        return []

    with open(inventory_path, "r", encoding="utf-8") as f:
        inventory = json.load(f)

    session_names = []
    for row in inventory:
        if "session_name" in row:
            session_names.append(normalize_session_name(row["session_name"]))

    session_names = sort_names(list(set(session_names)))
    return [str(x) for x in session_names]


def get_available_sessions(subject_id):
    """
    Session listesini bulur.

    Oncelik:
    1) Inventory json
    2) Dataset fallback
    """
    # Inventory varsa once onu kullanmak, daha once belgelenmis session listesini tekrarlar.
    # Dataset fallback ise inventory uretilmemis ortamlarda hattin calisabilmesi icindir.
    sessions = get_sessions_from_inventory(subject_id)
    if len(sessions) > 0:
        print("Session listesi inventory dosyasindan bulundu.")
        return sessions

    try:
        sessions = get_sessions_from_dataset(subject_id)
        if len(sessions) > 0:
            print("Session listesi dataset'ten bulundu (fallback).")
            return sessions
    except Exception as e:
        print(f"Uyari: Dataset'ten session listesi alinamadi -> {e}")

    raise ValueError("Session listesi bulunamadi.")


# =========================================================
# 4) SESSION ARALIGI
# =========================================================

def apply_session_range(session_names, start_session, end_session):
    """
    Session listesine basit start-end filtresi uygular.
    """
    if start_session is None or end_session is None:
        return session_names

    filtered = []
    for session_name in session_names:
        session_id = int(normalize_session_name(session_name))
        if start_session <= session_id <= end_session:
            filtered.append(str(session_id))

    filtered = sort_names(list(set(filtered)))
    filtered = [str(x) for x in filtered]

    if len(filtered) == 0:
        raise ValueError("Verilen aralikta islenecek session bulunamadi.")

    return filtered


# =========================================================
# 5) SCRIPT CALISTIRMA
# =========================================================

def run_one_script_for_session(script_name, session_name):
    """
    Verilen script'i tek bir session icin calistirir.
    """
    command = [sys.executable, script_name, str(session_name)]

    # Her adim ayri process olarak kosulur; boylece CLI tabanli eski script davranisi korunur.
    # Bir adim hatali donerse o session durur ve hata batch log'una yazilir.
    print(f"  -> Calisiyor: {script_name} (session {session_name})")
    result = subprocess.run(command, cwd=config.BASE_DIR, check=False)

    if result.returncode != 0:
        print(f"  !! Hata: {script_name} basarisiz (return code={result.returncode})")
        return False

    print(f"  OK: {script_name} tamamlandi.")
    return True


def run_pipeline_for_session(session_name):
    """
    Tek bir session icin tum feature uretim hattini calistirir.
    """
    for script_name in PIPELINE_SCRIPTS:
        ok = run_one_script_for_session(script_name, session_name)
        if not ok:
            return False, script_name

    return True, None


# =========================================================
# 6) OZET
# =========================================================

def print_summary(successful_sessions, failed_sessions):
    """
    Batch sonu ozetini ekrana basar.
    """
    total_processed = len(successful_sessions) + len(failed_sessions)

    print("\n===== BATCH OZET =====")
    print("Total processed:", total_processed)
    print("Successful sessions:", successful_sessions)
    print("Failed sessions:", [row["session"] for row in failed_sessions])

    if len(failed_sessions) > 0:
        print("\nBasarisiz session detaylari:")
        for row in failed_sessions:
            print(
                f"  session={row['session']} | "
                f"failed_step={row['failed_step']}"
            )


# =========================================================
# 7) LOG KAYDI
# =========================================================

def save_batch_log(subject_id, successful_sessions, failed_sessions):
    """
    Batch sonucunu kucuk bir json log dosyasina yazar.
    """
    # Batch log, hangi session'larin basariyla feature urettigini belgeleyen reproducibility ciktisidir.
    # Basarisiz adimlar acik tutuldugu icin eksik dosyalarin kaynagi sonradan izlenebilir.
    os.makedirs(config.LOG_DIR, exist_ok=True)

    save_path = os.path.join(
        config.LOG_DIR,
        f"batch_pipeline_subject_{subject_id}.json"
    )

    log_data = {
        "subject_id": int(subject_id),
        "successful_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    print(f"\nBatch log kaydedildi: {save_path}")


# =========================================================
# 8) ANA AKIS
# =========================================================

def main():
    subject_id, start_session, end_session = parse_cli_args()

    print(f"Batch pipeline basliyor. Subject: {subject_id}")

    sessions = get_available_sessions(subject_id)
    sessions = apply_session_range(sessions, start_session, end_session)
    print("Islenecek session'lar:", sessions)

    successful_sessions = []
    failed_sessions = []

    for session_name in sessions:
        print("\n----------------------------------------")
        print(f"Session {session_name} basladi.")

        ok, failed_step = run_pipeline_for_session(session_name)

        if ok:
            successful_sessions.append(str(session_name))
            print(f"Session {session_name} basariyla tamamlandi.")
        else:
            failed_sessions.append(
                {
                    "session": str(session_name),
                    "failed_step": failed_step
                }
            )
            print(f"Session {session_name} basarisiz. Diger session'a geciliyor.")

    print_summary(successful_sessions, failed_sessions)
    save_batch_log(subject_id, successful_sessions, failed_sessions)


if __name__ == "__main__":
    main()
