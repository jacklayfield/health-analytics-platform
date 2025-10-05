import dash
import dash_bootstrap_components as dbc
from visualization.layout import create_layout
from visualization import callbacks

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Health Data Explorer"
app.layout = create_layout()

server = app.server

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)