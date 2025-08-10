from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import date

def create_layout():
    return dbc.Container([
        html.H1("Health Data Explorer"),
        
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
            dbc.Col([
                dcc.Graph(id="main-chart")
            ], md=8),

            dbc.Col([
                html.H5("Data Table"),
                html.Div(id="data-table")
            ], md=4)
        ])
    ], fluid=True)
