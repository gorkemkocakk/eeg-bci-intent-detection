import json
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


MANIFEST_FILENAME = "manifest.json"
CAUTION_TEXT = "Generated outputs are local artifacts and should not be committed."


# Manifest, deney ciktisinin hangi kod, CLI argumani ve ortamla uretildigini belgelemek icindir.
# Bu dosya model performansini degil, tekrar uretilebilirlik bilgisini saklar.
def run_git_command(args):
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def get_git_info():
    # Git commit ve dirty bilgisi, ayni CSV sonucunun hangi kod durumundan geldigini anlamayi saglar.
    # Git komutu calismazsa deney akisi durmaz; alanlar None/unknown olarak yazilir.
    commit_hash = run_git_command(["rev-parse", "HEAD"])
    branch = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    status_short = run_git_command(["status", "--short"])

    if status_short is None:
        dirty = "unknown"
    else:
        dirty = bool(status_short)

    return {
        "commit_hash": commit_hash,
        "branch": branch,
        "dirty": dirty,
        "status_short": status_short,
    }


def make_json_safe(value):
    # Path ve tuple gibi JSON'un dogrudan yazamadigi tipler kayit oncesi sade tipe cevrilir.
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def write_manifest(
    output_dir,
    script_name,
    cli_args=None,
    input_dir=None,
    output_dir_value=None,
    extra=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Manifest alanlari, bir deney sonucunu geriye donuk takip etmek icin minimum iz birakir.
    # argv ve cli_args birlikte tutulur; biri ham komutu, digeri scriptin anlamlandirdigi degerleri gosterir.
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_name": script_name,
        "argv": sys.argv,
        "current_working_directory": str(Path.cwd()),
        "git": get_git_info(),
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cli_args": make_json_safe(cli_args or {}),
        "input_dir": make_json_safe(input_dir),
        "output_dir": make_json_safe(output_dir_value or output_path),
        "caution": CAUTION_TEXT,
    }

    if extra:
        # Script'e ozel ek bilgiler burada tutulur; ana manifest semasi bozulmadan genisletilebilir.
        manifest["extra"] = make_json_safe(extra)

    manifest_path = output_path / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Saved manifest: {manifest_path}")
    return manifest_path
