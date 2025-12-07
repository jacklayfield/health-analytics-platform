import json
import pandas as pd
import os
import logging
from typing import List, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _list_to_semicolon(value: Any) -> str:
    if isinstance(value, list):
        return ";".join([str(v).strip() for v in value if v is not None and str(v).strip()])
    if value is None:
        return ""
    return str(value)


def _read_input(input_path: str):
    """Read JSON/NDJSON/CSV input and return a list of records (dicts)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input synthea file not found: {input_path}")

    lower = input_path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(input_path).to_dict(orient="records")

    # Try JSON first
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        if not text:
            return []
        # NDJSON (line-delimited JSON)
        if '\n' in text and text.lstrip().startswith('{'):
            items = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    logger.exception("Failed to parse NDJSON line")
            return items
        # JSON array or single object
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            return [obj]
        except Exception:
            logger.exception("Failed to parse JSON input %s", input_path)
            return []


def transform_synthea_data(
    input_path: str = "/opt/airflow/data/raw/synthea/patients.json",
    output_path: str = "/opt/airflow/data/processed/synthea_patients.csv",
):
    """Simple transform that flattens Synthea patient records into a CSV.

    Produces columns:
      - patient_id, birth_date, gender, race, ethnicity, conditions, medications
    """
    logger.info("Transforming synthea data from %s...", input_path)

    data = _read_input(input_path)

    rows: List[dict] = []
    for rec in data:
        rows.append({
            "patient_id": rec.get("id") or rec.get("patientId"),
            "birth_date": rec.get("birthDate") or rec.get("birth_date"),
            "gender": rec.get("gender"),
            "race": rec.get("race"),
            "ethnicity": rec.get("ethnicity"),
            "conditions": _list_to_semicolon(rec.get("conditions") or rec.get("conditions_list") or rec.get("conditions[]")),
            "medications": _list_to_semicolon(rec.get("medications") or rec.get("meds") or rec.get("medications[]")),
        })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Wrote %d patient rows to %s", len(df), output_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Transform synthea JSON -> processed CSV")
    p.add_argument("--input", default="/opt/airflow/data/raw/synthea/patients.json")
    p.add_argument("--output", default="/opt/airflow/data/processed/synthea_patients.csv")
    args = p.parse_args()
    transform_synthea_data(input_path=args.input, output_path=args.output)
