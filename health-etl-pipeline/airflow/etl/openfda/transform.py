import json
import pandas as pd
import os

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
                    "reaction": reaction_name,
                    "medicinalproduct": drug.get("medicinalproduct"),
                    "drugauthorizationnumb": drug.get("drugauthorizationnumb"),
                    "brand_name": ", ".join(openfda.get("brand_name", [])),
                    "manufacturer_name": ", ".join(openfda.get("manufacturer_name", [])),
                    "product_ndc": ", ".join(openfda.get("product_ndc", []))
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Transformed data saved to {output_path}")
