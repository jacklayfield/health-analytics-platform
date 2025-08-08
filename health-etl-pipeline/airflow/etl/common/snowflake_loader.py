from airflow.etl.common.base_loader import DataLoader
import pandas as pd

class SnowflakeLoader(DataLoader):
    def load(self, df: pd.DataFrame):
        print(f"[Mock] Will load {len(df)} rows into Snowflake.")
        print(df.head(3)) 
