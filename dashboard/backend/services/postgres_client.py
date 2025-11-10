import pandas as pd
import sqlalchemy
from config import POSTGRES_URI

engine = sqlalchemy.create_engine(POSTGRES_URI)

def query_df(sql: str, params=None, limit=None) -> pd.DataFrame:
    if limit:
        sql = f"{sql} LIMIT {limit}"
    return pd.read_sql(sql, engine, params=params)

def trends_by_date(table: str, date_col: str="date", metric_col: str="value", group_by: str="city", limit=1000):
    sql = f"""
      SELECT {group_by} as group_key, {date_col}::date as date, AVG({metric_col}) as avg_value
      FROM {table}
      GROUP BY group_key, {date_col}::date
      ORDER BY date DESC
      LIMIT {limit}
    """
    return pd.read_sql(sql, engine)