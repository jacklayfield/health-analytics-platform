import pandas as pd
from etl.common.postgres_loader import PostgresLoader
from etl.common.snowflake_loader import SnowflakeLoader
import os

def load_openfda_data(
    input_path: str = "/opt/airflow/data/processed/events.csv",
    warehouse: str = "postgres",
):
    """
    Read the transformed CSV and load it into the specified warehouse.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Expected transformed file not found: {input_path}")

    df = pd.read_csv(input_path)

    if warehouse == "postgres":
        db_uri: str = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
        loader = PostgresLoader(db_uri=db_uri, table_name="openfda_events")
    elif warehouse == "snowflake":
        loader = SnowflakeLoader()
    else:
        raise ValueError(f"Unknown warehouse: {warehouse}")

    loader.load(df)
