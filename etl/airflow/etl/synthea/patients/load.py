import pandas as pd
import os
import logging
from etl.common.postgres_loader import PostgresLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_synthea_data(
    input_path: str = "/opt/airflow/data/processed/synthea_patients.csv",
    warehouse: str = "postgres",
    db_uri: str | None = None,
):
    """Load the processed CSV into the target warehouse.

    By default will use the `WAREHOUSE_DB_URI` environment variable if present,
    falling back to a sensible Docker Compose Postgres URI used by the ETL
    compose files.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Transformed file not found: {input_path}")

    df = pd.read_csv(input_path)

    if db_uri is None:
        db_uri = os.environ.get(
            "WAREHOUSE_DB_URI",
            "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow",
        )

    if warehouse != "postgres":
        raise ValueError(f"Unsupported warehouse: {warehouse}")

    logger.info("Loading %d rows into warehouse via %s", len(df), db_uri)
    loader = PostgresLoader(db_uri=db_uri, table_name="synthea_patients")
    loader.load(df)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Load processed synthea CSV into warehouse")
    p.add_argument("--input", default="/opt/airflow/data/processed/synthea_patients.csv")
    p.add_argument("--warehouse", default="postgres")
    p.add_argument("--db-uri", default=None)
    args = p.parse_args()
    load_synthea_data(input_path=args.input, warehouse=args.warehouse, db_uri=args.db_uri)
