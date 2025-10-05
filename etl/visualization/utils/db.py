import psycopg2
import pandas as pd

# Note: need to rework when using Snowflake
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="your_db",
        user="your_user",
        password="your_password"
    )

def query_data(dataset, start_date, end_date):
    conn = get_connection()
    query = f"""
        SELECT date, value
        FROM {dataset}
        WHERE date BETWEEN %s AND %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(start_date, end_date))
    conn.close()
    return df
