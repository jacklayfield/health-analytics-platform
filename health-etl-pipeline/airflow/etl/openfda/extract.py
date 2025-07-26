import os
import requests
import json

def download_openfda_data(api_url: str = "https://api.fda.gov/drug/event.json", output_path: str = "/opt/airflow/data/raw/events.json", limit: int = 100):
    print(f"Fetching openFDA data from {api_url}...")

    params = {"limit": limit}
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved openFDA API data to {output_path}")

    except requests.exceptions.HTTPError as errh:
        raise Exception(f"HTTP Error: {errh}")
    except requests.exceptions.RequestException as err:
        raise Exception(f"Request failed: {err}")