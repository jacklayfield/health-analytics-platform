import json
from datetime import datetime
from ..utils import csv_to_records

def transform_patients(**kwargs):
    """Transform patients data from CSV."""
    csv_path = 'data/csv/patients.csv'
    records = csv_to_records(csv_path)
    
    for record in records:
        if record.get('birthDate'):
            record['birthDate'] = datetime.fromisoformat(record['birthDate'])
    
    processed_path = 'data/processed/patients.json'
    with open(processed_path, 'w') as f:
        json.dump(records, f, default=str)
    
    print(f"Transformed {len(records)} patient records")
