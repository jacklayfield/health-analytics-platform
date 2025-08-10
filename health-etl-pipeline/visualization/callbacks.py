from dash import Input, Output, dash_table
from app import app
from utils.db import query_data
import plotly.express as px

@app.callback(
    Output("main-chart", "figure"),
    Output("data-table", "children"),
    Input("dataset-dropdown", "value"),
    Input("date-range-picker", "start_date"),
    Input("date-range-picker", "end_date")
)
def update_outputs(dataset, start_date, end_date):
    df = query_data(dataset, start_date, end_date)
    
    # Chart
    fig = px.line(df, x="date", y="value", title=f"{dataset} Trends")

    # Table
    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=10
    )

    return fig, table