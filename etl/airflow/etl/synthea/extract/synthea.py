from pathlib import Path

def extract_synthea(**kwargs):
    """Extract step: Check if data exists (generated externally)."""
    data_dir = Path("/opt/airflow/data")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise RuntimeError("No data found in /opt/airflow/data. Please generate data externally using 'python run_synthea.py -p 100' on the host.")
    print("Data found, extraction complete.")