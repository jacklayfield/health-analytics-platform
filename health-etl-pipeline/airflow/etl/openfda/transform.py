import json
import pandas as pd
import os

def clean_first_value(value):
    """Safely extract and clean the first item of a list, if present."""
    if isinstance(value, list) and value:
        return value[0].strip()
    if isinstance(value, str):
        return value.strip()
    return None

def transform_openfda_data(
    input_path="/opt/airflow/data/raw/events.json",
    output_path="/opt/airflow/data/processed/events.csv"
):
    print(f"Transforming data from {input_path}...")

    with open(input_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    rows = []

    for report in results:
        base = {
            "safetyreportid": report.get("safetyreportid"),
            "receivedate": report.get("receivedate"),
            "serious": report.get("serious"),
            "patientonsetage": report.get("patient", {}).get("patientonsetage"),
            "patientsex": report.get("patient", {}).get("patientsex")
        }

        reactions = report.get("patient", {}).get("reaction", [])
        drugs = report.get("patient", {}).get("drug", [])

        for reaction in reactions:
            reaction_name = reaction.get("reactionmeddrapt")

            for drug in drugs:
                openfda = drug.get("openfda", {})

                rows.append({
                    **base,
                    "reaction": reaction_name.strip() if reaction_name else None,
                    "medicinalproduct": drug.get("medicinalproduct", "").strip(),
                    "drugauthorizationnumb": drug.get("drugauthorizationnumb", "").strip(),
                    "brand_name": clean_first_value(openfda.get("brand_name")),
                    "manufacturer_name": clean_first_value(openfda.get("manufacturer_name")),
                    "product_ndc": clean_first_value(openfda.get("product_ndc"))
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Transformed data saved to {output_path}")
