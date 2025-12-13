import json
import csv
from pathlib import Path
import sys

# -------------------------
# Resolve paths safely
# -------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

SYNTHEA_DIR = (SCRIPT_DIR / "../../../../../synthea").resolve()
FHIR_DIR = SYNTHEA_DIR / "output" / "fhir"

OUT_DIR = SCRIPT_DIR / "data" / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[fhir_to_csv] Script dir: {SCRIPT_DIR}")
print(f"[fhir_to_csv] Synthea dir: {SYNTHEA_DIR}")
print(f"[fhir_to_csv] FHIR dir: {FHIR_DIR}")

# -------------------------
# Validate input
# -------------------------

fhir_files = list(FHIR_DIR.glob("*.json"))

if not fhir_files:
    print("[fhir_to_csv] ERROR: No FHIR JSON files found.")
    sys.exit(1)

print(f"[fhir_to_csv] Found {len(fhir_files)} FHIR bundles")

# -------------------------
# Data buckets
# -------------------------

patients = []
encounters = []
conditions = []
observations = []
medications = []

def ref(r):
    return r.get("reference") if isinstance(r, dict) else None

# -------------------------
# Parse FHIR bundles
# -------------------------

for f in fhir_files:
    with f.open(encoding="utf-8") as fh:
        bundle = json.load(fh)

    for entry in bundle.get("entry", []):
        r = entry.get("resource", {})
        rt = r.get("resourceType")

        if rt == "Patient":
            patients.append({
                "id": r.get("id"),
                "gender": r.get("gender"),
                "birthDate": r.get("birthDate")
            })

        elif rt == "Encounter":
            encounters.append({
                "id": r.get("id"),
                "patient": ref(r.get("subject")),
                "start": r.get("period", {}).get("start"),
                "end": r.get("period", {}).get("end")
            })

        elif rt == "Condition":
            conditions.append({
                "id": r.get("id"),
                "patient": ref(r.get("subject")),
                "code": r.get("code", {}).get("text")
            })

        elif rt == "Observation":
            observations.append({
                "id": r.get("id"),
                "patient": ref(r.get("subject")),
                "code": r.get("code", {}).get("text"),
                "value": r.get("valueQuantity", {}).get("value")
            })

        elif rt == "MedicationRequest":
            medications.append({
                "id": r.get("id"),
                "patient": ref(r.get("subject")),
                "medication": r.get("medicationCodeableConcept", {}).get("text")
            })

# -------------------------
# Write CSVs
# -------------------------

def write(name, rows):
    if not rows:
        print(f"[fhir_to_csv] No rows for {name}, skipping")
        return

    out = OUT_DIR / f"{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"[fhir_to_csv] Wrote {out} ({len(rows)} rows)")

write("patients", patients)
write("encounters", encounters)
write("conditions", conditions)
write("observations", observations)
write("medications", medications)

print("[fhir_to_csv] DONE")
