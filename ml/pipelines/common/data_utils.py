# Helper functions to connect to the warehouse and fetch OpenFDA data.

import os
import pandas as pd
from sqlalchemy import create_engine


def get_engine() -> 'sqlalchemy.Engine':
    db_uri = os.getenv(
        "WAREHOUSE_DB_URI",
        "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
    )
    return create_engine(db_uri)


def load_openfda_events(columns: list[str] | None = None) -> pd.DataFrame:
    engine = get_engine()

    col_clause = ", ".join(columns) if columns else "*"
    query = f"SELECT {col_clause} FROM openfda_events;"
    df = pd.read_sql(query, engine)

    if df.empty:
        raise RuntimeError("openfda_events query returned no rows—check DB/connection.")

    if "serious" in df.columns:
        distinct_classes = df["serious"].dropna().unique()
        if len(distinct_classes) < 2:
            raise RuntimeError(
                f"'serious' column has fewer than two classes: {distinct_classes}. "
                "Check database contents or connection string."
            )

    return df
