import dash
from dash import dcc, html
from layouts import create_layout
import callbacks

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Health Data Explorer"

app.layout = create_layout()

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
