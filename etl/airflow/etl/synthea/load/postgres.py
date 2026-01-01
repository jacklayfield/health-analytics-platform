import pandas as pd
import os
from ...common.postgres_loader import PostgresLoader

def load_patients(**kwargs):
    """Load patients data into PostgreSQL."""
    processed_path = '/opt/airflow/data/processed/patients.json'
    df = pd.read_json(processed_path)
    
    db_uri = os.getenv('POSTGRES_URI', 'postgresql://airflow:airflow@localhost:5432/airflow')
    loader = PostgresLoader(db_uri=db_uri, table_name='patients')
    loader.load(df)
    
    print(f"Loaded patients data into database")
