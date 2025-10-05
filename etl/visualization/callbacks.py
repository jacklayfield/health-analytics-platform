from dash import Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import psycopg2
from warehouse import queries
from visualization.app import app

def run_query(query, params=None):
    conn = psycopg2.connect(
        dbname="warehouse",
        user="airflow",
        password="airflow",
        host="postgres",
        port="5432"
    )
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

@app.callback(
    [
        Output("main-chart", "figure"),
        Output("data-table", "children"),
        Output("category-chart", "figure"),
        Output("trend-chart", "figure"),
        Output("kpi-total", "children"),
        Output("kpi-unique", "children"),
        Output("kpi-latest", "children"),
    ],
    [
        Input("dataset-dropdown", "value"),
        Input("date-range-picker", "start_date"),
        Input("date-range-picker", "end_date"),
    ]
)
def update_visuals(dataset, start_date, end_date):
    if dataset == "openfda":
        df = run_query(queries.openfda_time_series, (start_date, end_date))
        df_cat = run_query(queries.openfda_by_category, (start_date, end_date))
        df_trend = df.copy()
    elif dataset == "synthea":
        df = run_query(queries.synthea_time_series, (start_date, end_date))
        df_cat = run_query(queries.synthea_by_category, (start_date, end_date))
        df_trend = df.copy()
    else:
        return go.Figure(), "", go.Figure(), go.Figure(), "", "", ""

    main_chart = px.line(
        df,
        x="date",
        y="count",
        title=f"{dataset.capitalize()} Records Over Time"
    )

    category_chart = px.bar(
        df_cat,
        x="category",
        y="count",
        title=f"{dataset.capitalize()} by Category"
    )

    # Trend chart
    df_trend["rolling_avg"] = df_trend["count"].rolling(window=7).mean()
    trend_chart = go.Figure()
    trend_chart.add_trace(go.Scatter(x=df_trend["date"], y=df_trend["count"],
                                     mode="lines", name="Daily"))
    trend_chart.add_trace(go.Scatter(x=df_trend["date"], y=df_trend["rolling_avg"],
                                     mode="lines", name="7-day Avg"))
    trend_chart.update_layout(title=f"{dataset.capitalize()} Trend")

    # Data table
    data_table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict("records"),
        page_size=10,
        style_table={"overflowX": "auto"},
    )

    # KPIs
    total_records = f"{df['count'].sum():,}"
    unique_dates = df["date"].nunique()
    latest_date = df["date"].max()

    return main_chart, data_table, category_chart, trend_chart, total_records, unique_dates, str(latest_date)
