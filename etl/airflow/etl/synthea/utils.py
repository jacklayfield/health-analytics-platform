import csv
import json
import logging
from typing import List

logger = logging.getLogger(__name__)


def csv_to_records(path: str) -> List[dict]:
    """Read a CSV file and return list of dict records.

    This function uses the stdlib `csv` module so it works without pandas, but
    if pandas is available it will be used for better dtype handling.
    """
    try:
        import pandas as pd

        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception:
        logger.debug("pandas not available or failed; falling back to csv module for %s", path)

    records = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # convert empty strings to None? keep as-is for now
            records.append({k: (v if v != '' else None) for k, v in row.items()})
    return records


def write_json(path: str, records: List[dict]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
