import pandas as pd
from etl.common.postgres_loader import PostgresLoader
from etl.common.snowflake_loader import SnowflakeLoader

def load_openfda_data(df: pd.DataFrame, warehouse: str = "postgres"):
    """
    Load transformed OpenFDA data into the specified warehouse.
    
    Args:
        df: pandas DataFrame containing transformed OpenFDA data
        warehouse: "postgres" or "snowflake"
    """
    if warehouse == "postgres":
        loader = PostgresLoader()
    elif warehouse == "snowflake":
        loader = SnowflakeLoader()
    else:
        raise ValueError(f"Unknown warehouse: {warehouse}")

    loader.load_dataframe(df, table_name="openfda_events")
