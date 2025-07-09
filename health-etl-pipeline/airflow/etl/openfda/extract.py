import os
import requests
import zipfile

DATA_DIR = "/opt/airflow/data/raw/openfda/"
ZIP_URL = "https://download.open.fda.gov/drug/event/drug-event-0001-of-0001.json.zip"

def download_openfda_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "drug-event-latest.zip")

    print(f"Downloading from {ZIP_URL}...")
    response = requests.get(ZIP_URL, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print(f"Data extracted to {DATA_DIR}")
