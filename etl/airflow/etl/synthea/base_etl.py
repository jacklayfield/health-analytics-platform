import os
import json
import shutil
import logging
from typing import Optional, List

from etl.airflow.etl.synthea.utils import csv_to_records

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _read_json_or_ndjson(path: str) -> List[dict]:
    """Read a JSON file (array or object) or NDJSON and return a list of records."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        if not text:
            return []
        # NDJSON heuristics: multiple lines that start with '{'
        if '\n' in text and text.lstrip().startswith('{'):
            items = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    logger.exception("Failed to parse NDJSON line in %s", path)
            return items
        # JSON array or single object
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        return [obj]


def _convert_candidate_to_json(path: str) -> List[dict]:
    lower = path.lower()
    if lower.endswith('.json') or lower.endswith('.ndjson') or lower.endswith('.ndjsonl'):
        return _read_json_or_ndjson(path)
    if lower.endswith('.csv'):
        return csv_to_records(path)
    raise ValueError(f"Unsupported candidate file type: {path}")


def prepare_resource(
    resource: str,
    input_dir: str = "/opt/airflow/data/raw/synthea",
    output_path: Optional[str] = None,
) -> str:
    """Normalize a resource from the raw Synthea output into a JSON array file.

    - If `output_path` exists already, returns it.
    - If `input_dir/<resource>/` exists, finds the first candidate file and converts it
      into `input_dir/<resource>.json`.
    - If `input_dir/<resource>.json` already exists, uses it.
    - If no files are found and resource == 'patients', writes a small sample.

    Returns the path to the normalized JSON file.
    """
    if output_path is None:
        output_path = os.path.join(input_dir, f"{resource}.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        logger.info("Resource JSON already present at %s", output_path)
        return output_path

    # Look for resource subdirectory
    subdir = os.path.join(input_dir, resource)
    if os.path.isdir(subdir):
        for fname in sorted(os.listdir(subdir)):
            cand = os.path.join(subdir, fname)
            if not os.path.isfile(cand):
                continue
            try:
                records = _convert_candidate_to_json(cand)
                with open(output_path, 'w', encoding='utf-8') as dst:
                    json.dump(records, dst)
                logger.info("Normalized %s -> %s", cand, output_path)
                return output_path
            except Exception:
                logger.exception("Failed to convert candidate %s", cand)

    # Fallback: look for resource file at raw root
    candidates = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
    for fname in candidates:
        if fname.lower() in (f"{resource}.json", f"{resource}.ndjson", f"{resource}.csv"):
            path = os.path.join(input_dir, fname)
            try:
                records = _convert_candidate_to_json(path)
                with open(output_path, 'w', encoding='utf-8') as dst:
                    json.dump(records, dst)
                logger.info("Normalized %s -> %s", path, output_path)
                return output_path
            except Exception:
                logger.exception("Failed to convert root candidate %s", path)

    # Nothing found — for patients create a small sample, otherwise create empty list
    if resource == 'patients':
        sample = [
            {"id": "patient-1", "birthDate": "1970-01-01", "gender": "female"},
            {"id": "patient-2", "birthDate": "1985-05-12", "gender": "male"},
        ]
        with open(output_path, 'w', encoding='utf-8') as dst:
            json.dump(sample, dst, indent=2)
        logger.info("Wrote sample patients to %s", output_path)
        return output_path

    # create an empty JSON array for other resources
    with open(output_path, 'w', encoding='utf-8') as dst:
        json.dump([], dst)
    logger.info("Wrote empty resource file %s", output_path)
    return output_path
