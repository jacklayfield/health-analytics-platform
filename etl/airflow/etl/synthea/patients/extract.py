from typing import Optional

from etl.airflow.etl.synthea.base_etl import prepare_resource


def download_synthea_patients(
    input_dir: Optional[str] = "/opt/airflow/data/raw/synthea",
    output_path: Optional[str] = None,
):
    """Wrapper to prepare the `patients` resource using shared base_etl logic."""
    return prepare_resource("patients", input_dir=input_dir, output_path=output_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Prepare synthea patients for ETL pipeline")
    p.add_argument("--input-dir", default="/opt/airflow/data/raw/synthea")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    path = download_synthea_patients(input_dir=args.input_dir, output_path=args.output)
    print(path)
