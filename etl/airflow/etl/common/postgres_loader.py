from sqlalchemy import create_engine
import pandas as pd
from etl.common.base_loader import DataLoader

class PostgresLoader(DataLoader):
    def __init__(self, db_uri, table_name):
        self.db_uri = db_uri
        self.table_name = table_name

    def load(self, df: pd.DataFrame):
        engine = create_engine(self.db_uri)
        with engine.begin() as conn:
            df.to_sql(self.table_name, conn, if_exists="replace", index=False)
        print(f"Loaded {len(df)} rows into {self.table_name}")