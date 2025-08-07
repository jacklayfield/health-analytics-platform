import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


def load_openfda_data_to_postgres(
    csv_path: str = "/opt/airflow/data/processed/events.csv",
    db_uri: str = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow",
    table_name: str = "openfda_events"
) -> None:
    """
    Load processed OpenFDA CSV data into a PostgreSQL table using SQLAlchemy.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("CSV file is empty. Aborting load.")

        engine = create_engine(db_uri)

        with engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        
        print(f"Successfully loaded {len(df)} records into '{table_name}'.")

    except (SQLAlchemyError, ValueError, pd.errors.ParserError) as e:
        raise RuntimeError(f"Failed to load data into Postgres: {e}")
