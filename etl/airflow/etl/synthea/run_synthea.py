import os
import shutil
import subprocess
import platform
from pathlib import Path

# =========================
# CONFIG
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent

SYNTHEA_DIR = Path(
    os.environ.get(
        "SYNTHEA_DIR",
        SCRIPT_DIR / "../../../../../synthea"
    )
).resolve()

OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        SCRIPT_DIR / "data"
    )
).resolve()

NUM_PATIENTS = os.environ.get("NUM_PATIENTS", "100")
JAVA_CMD = os.environ.get("JAVA_CMD", "java")

# =========================
# HELPERS
# =========================

def log(msg):
    print(f"\n[run_synthea] {msg}")

# =========================
# CLONE IF MISSING
# =========================

if not SYNTHEA_DIR.exists():
    log("Cloning Synthea...")
    subprocess.run(
        ["git", "clone", "https://github.com/synthetichealth/synthea.git", str(SYNTHEA_DIR)],
        check=True
    )

# =========================
# ENABLE EXPORTERS
# =========================

props_file = SYNTHEA_DIR / "src/main/resources/synthea.properties"

log("Ensuring CSV, FHIR, and CCDA exporters are enabled...")

props = props_file.read_text(encoding="utf-8").splitlines()
props_dict = {}

for line in props:
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        props_dict[k.strip()] = v.strip()

props_dict["exporter.csv"] = "true"
props_dict["exporter.fhir"] = "true"
props_dict["exporter.ccda"] = "true"

with props_file.open("w", encoding="utf-8") as f:
    for k, v in sorted(props_dict.items()):
        f.write(f"{k}={v}\n")

# =========================
# BUILD SYNTHEA
# =========================

log("Building Synthea...")

if platform.system() == "Windows":
    gradle_cmd = ["gradlew.bat"]
    shell = True
else:
    gradle_cmd = ["./gradlew"]
    shell = False

subprocess.run(
    gradle_cmd + ["build", "-x", "test"],
    cwd=SYNTHEA_DIR,
    check=True,
    shell=shell
)

# =========================
# RUN SYNTHEA
# =========================

jar_path = SYNTHEA_DIR / "build/libs/synthea-with-dependencies.jar"

if not jar_path.exists():
    raise FileNotFoundError(f"Synthea jar not found: {jar_path}")

log("Running Synthea (nationwide, CSV + FHIR)...")

subprocess.run(
    [
        JAVA_CMD,
        "-jar",
        str(jar_path),
        "-p",
        str(NUM_PATIENTS)
    ],
    cwd=SYNTHEA_DIR,
    check=True
)

# =========================
# COPY OUTPUTS
# =========================

log("Copying output files...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for subdir in ["csv", "fhir", "ccda"]:
    src = SYNTHEA_DIR / "output" / subdir
    dst = OUTPUT_DIR / subdir

    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        log(f"Copied {subdir} to {dst}")
    else:
        log(f"No {subdir} output found, skipping.")

log(f"Done! Outputs located at: {OUTPUT_DIR}")
