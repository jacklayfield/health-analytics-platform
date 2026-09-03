import os
from pathlib import Path

import pandas as pd

from etl.common.postgres_loader import PostgresLoader
from etl.common.snowflake_loader import SnowflakeLoader


def load_cms_desynpuf(
    input_root: str = "/opt/airflow/data/processed/cms_desynpuf",
    warehouse: str = "postgres",
    # Chunking due to large data set (over 1 million rows)
    chunk_size: int = 10_000,
) -> dict[str, int]:
    """Load transformed CMS DE-SynPUF datasets into the selected warehouse."""
    input_dir = Path(input_root)
    input_paths = sorted(input_dir.glob("*.csv"))

    if not input_paths:
        raise FileNotFoundError(
            f"No transformed CMS CSV files found under {input_dir}"
        )

    if warehouse not in {"postgres", "snowflake"}:
        raise ValueError(f"Unknown warehouse: {warehouse}")

    db_uri = os.getenv(
        "POSTGRES_URI",
        "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow",
    )
    loaded_rows = {}

    for input_path in input_paths:
        table_name = f"cms_desynpuf_{input_path.stem}"
        loaded_rows[table_name] = 0
        first_chunk = True

        for df in pd.read_csv(input_path, dtype=str, chunksize=chunk_size):
            if warehouse == "postgres":
                loader = PostgresLoader(db_uri=db_uri, table_name=table_name)
                loader.load(df, if_exists="replace" if first_chunk else "append")
            else:
                loader = SnowflakeLoader()
                loader.load(df)

            loaded_rows[table_name] += len(df)
            first_chunk = False

    return loaded_rows