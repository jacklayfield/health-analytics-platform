import pandas as pd
import sqlalchemy
from config import POSTGRES_URI

engine = sqlalchemy.create_engine(POSTGRES_URI)

def query_df(sql: str, params=None, limit=None) -> pd.DataFrame:
    if limit:
        sql = f"{sql} LIMIT {limit}"
    return pd.read_sql(sql, engine, params=params)

def trends_by_date(table: str, date_col: str="date", metric_col: str="value", group_by: str="city", limit=1000):
    # Handle different date formats
    if date_col == "receivedate":
        # receivedate is stored as integer in YYYYMMDD format
        date_expr = f"to_date({date_col}::text, 'YYYYMMDD')"
    else:
        # Default assumption: date column can be cast to date
        date_expr = f"{date_col}::date"

    sql = f"""
      SELECT {group_by} as group_key, {date_expr} as date, AVG({metric_col}) as avg_value
      FROM {table}
      GROUP BY group_key, {date_expr}
      ORDER BY date DESC
      LIMIT {limit}
    """
    return pd.read_sql(sql, engine)