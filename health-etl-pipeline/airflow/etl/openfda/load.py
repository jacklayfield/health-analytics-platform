import pandas as pd
from sqlalchemy import create_engine
import os

def load_openfda_data_to_postgres(
    csv_path="/opt/airflow/data/processed/events.csv",
    db_uri="postgresql+psycopg2://airflow:airflow@postgres:5432/airflow",
    table_name="openfda_events"
):
    print(f"Loading data from {csv_path} into {table_name}...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist")

    # Read transformed CSV
    df = pd.read_csv(csv_path)

    # Create SQLAlchemy engine
    engine = create_engine(db_uri)

    # Load into DB
    with engine.begin() as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"✅ Loaded {len(df)} rows into {table_name}")
