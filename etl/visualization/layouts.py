from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import date

def create_layout():
    return dbc.Container([
        html.H1("Health Data Explorer", className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.Label("Dataset"),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=[
                        {"label": "OpenFDA", "value": "openfda"},
                        {"label": "Synthea", "value": "synthea"}
                    ],
                    value="openfda",
                    clearable=False
                )
            ], md=3),

            dbc.Col([
                html.Label("Date Range"),
                dcc.DatePickerRange(
                    id="date-range-picker",
                    start_date=date(2023, 1, 1),
                    end_date=date.today()
                )
            ], md=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Total Records", className="card-title"),
                    html.H3(id="kpi-total")
                ])
            ], className="shadow-sm"), md=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Unique Categories", className="card-title"),
                    html.H3(id="kpi-unique")
                ])
            ], className="shadow-sm"), md=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Latest Record Date", className="card-title"),
                    html.H3(id="kpi-latest")
                ])
            ], className="shadow-sm"), md=3),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id="main-chart")
            ], md=8),

            dbc.Col([
                html.H5("Data Table"),
                html.Div(id="data-table")
            ], md=4)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H5("Breakdown by Category"),
                dcc.Graph(id="category-chart")
            ], md=6),

            dbc.Col([
                html.H5("Trend Over Time"),
                dcc.Graph(id="trend-chart")
            ], md=6),
        ])
    ], fluid=True)
