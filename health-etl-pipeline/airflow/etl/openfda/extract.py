import os
import requests
import json
import time

def download_openfda_data(
    api_url="https://api.fda.gov/drug/event.json",
    output_path="/opt/airflow/data/raw/events.json",
    limit=100,
    max_records=1000
):
    print(f"Fetching openFDA data from {api_url}...")

    all_results = []
    skip = 0

    while True:
        params = {"limit": limit, "skip": skip}
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        page = response.json()

        results = page.get("results", [])
        if not results:
            break

        all_results.extend(results)

        if max_records and len(all_results) >= max_records:
            break

        skip += limit

        # Sleep to avoid setting off rquests per second violations
        time.sleep(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    print(f"Saved {len(all_results)} records to {output_path}")