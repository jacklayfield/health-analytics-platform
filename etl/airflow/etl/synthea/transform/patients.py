import json
import csv
from datetime import datetime
from pathlib import Path
from ..utils import csv_to_records

def _convert_fhir_to_csv():
    """Convert FHIR bundles to CSV if CSV doesn't exist."""
    csv_path = Path('/opt/airflow/data/csv/patients.csv')
    if csv_path.exists():
        return 
    
    script_dir = Path(__file__).resolve().parent.parent
    synthea_dir = Path("/opt/airflow/data")
    fhir_dir = synthea_dir / "fhir"

    if not fhir_dir.exists():
        raise FileNotFoundError(f"FHIR directory not found: {fhir_dir}")

    fhir_files = list(fhir_dir.glob("*.json"))
    if not fhir_files:
        raise FileNotFoundError(f"No FHIR JSON files found in {fhir_dir}")

    patients = []
    for f in fhir_files:
        with f.open(encoding="utf-8") as fh:
            bundle = json.load(fh)
        for entry in bundle.get("entry", []):
            r = entry.get("resource", {})
            if r.get("resourceType") == "Patient":
                patients.append({
                    "id": r.get("id"),
                    "gender": r.get("gender"),
                    "birthDate": r.get("birthDate")
                })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if patients:
            w = csv.DictWriter(f, fieldnames=patients[0].keys())
            w.writeheader()
            w.writerows(patients)

def transform_patients(**kwargs):
    """Transform patients data from CSV."""
    _convert_fhir_to_csv()
    
    csv_path = 'data/csv/patients.csv'
    records = csv_to_records(csv_path)
    
    for record in records:
        if record.get('birthDate'):
            record['birthDate'] = datetime.fromisoformat(record['birthDate'])
    
    processed_path = Path('/opt/airflow/data/processed/patients.json')
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, 'w') as f:
        json.dump(records, f, default=str)
    
    print(f"Transformed {len(records)} patient records")
