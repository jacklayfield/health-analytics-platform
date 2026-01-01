import os
import shutil
import subprocess
import platform
import sys
import argparse
import urllib.request
from pathlib import Path

# =========================
# ARGUMENTS
# =========================

parser = argparse.ArgumentParser(description="Run Synthea deterministically")

parser.add_argument(
    "-p", "--patients",
    type=int,
    default=int(os.environ.get("NUM_PATIENTS", 5)),
    help="Number of patients to generate"
)

parser.add_argument(
    "-s", "--seed",
    type=int,
    help="Random seed (reproducible runs)"
)

parser.add_argument(
    "-g", "--gender",
    choices=["M", "F"],
    help="Gender (M or F)"
)

parser.add_argument(
    "-a", "--age",
    help="Age range (e.g. 18-65)"
)

parser.add_argument(
    "-m", "--module",
    help="Path to a Synthea module JSON file"
)

parser.add_argument(
    "--rebuild",
    action="store_true",
    help="Force rebuild of Synthea"
)

# Positional arguments: state and optional city
parser.add_argument(
    "state",
    nargs="?",
    help="US state (e.g. 'Massachusetts', 'California')"
)
parser.add_argument(
    "city",
    nargs="?",
    help="City name (optional)"
)

args = parser.parse_args()

# =========================
# CONFIG
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent

SYNTHEA_DIR = SCRIPT_DIR

OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        SCRIPT_DIR / "../../../data"
    )
).resolve()

JAVA_CMD = os.environ.get("JAVA_CMD", "java")

# =========================
# HELPERS
# =========================

def log(msg):
    print(f"\n[run_synthea] {msg}")

def run(cmd, cwd=None, shell=False):
    subprocess.run(cmd, cwd=cwd, check=True, shell=shell)

# =========================
# DOWNLOAD JAR IF MISSING
# =========================

jar_path = SCRIPT_DIR / "synthea-with-dependencies.jar"

if not jar_path.exists():
    log("Downloading Synthea jar...")
    urllib.request.urlretrieve(
        "https://github.com/synthetichealth/synthea/releases/download/v3.1.0/synthea-with-dependencies.jar",
        jar_path
    )

# =========================
# CLEAN OUTPUT
# =========================

synthea_output = SCRIPT_DIR / "output"

if synthea_output.exists():
    log("Removing previous Synthea output...")
    shutil.rmtree(synthea_output)

# =========================
# BUILD COMMAND
# =========================

cmd = [
    JAVA_CMD,
    "-jar",
    str(jar_path),
    "-p",
    str(args.patients),
]

# Optional parameters
if args.seed is not None:
    cmd += ["-s", str(args.seed)]

if args.gender:
    cmd += ["-g", args.gender]

if args.age:
    cmd += ["-a", args.age]

if args.module:
    cmd += ["-m", args.module]

# Positional arguments: state [city]
if args.state:
    cmd.append(args.state)
    if args.city:
        cmd.append(args.city)

# =========================
# RUN SYNTHEA
# =========================

log("Running Synthea with command:")
log(" ".join(cmd))

run(cmd, cwd=SCRIPT_DIR)

# =========================
# COPY OUTPUTS
# =========================

log("Copying output files...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if synthea_output.exists():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(synthea_output, OUTPUT_DIR)
    log(f"Copied outputs → {OUTPUT_DIR}")
else:
    log("No output directory found, nothing to copy.")

log(f"Done! Outputs located at: {OUTPUT_DIR}")
