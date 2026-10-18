"""
The Dash application.

Served at the routes in ``DASH_ROUTES_PATHNAME_PREFIX`` behind a gateway that adds
``DASH_REQUESTS_PATHNAME_PREFIX``; Dash reads both from the environment.
"""

from pathlib import Path

import dash
from dash import dcc, html

HERE = Path(__file__).parent

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(HERE / "pages"),
    assets_folder=str(HERE.parent / "assets"),
    title="neurocorr",
    suppress_callback_exceptions=True,
)
server = app.server


def navigation():
    return html.Nav(
        [html.Span("neurocorr", className="brand")]
        + [
            dcc.Link(page["name"], href=page["relative_path"], className="nav-link")
            for page in dash.page_registry.values()
            if not page.get("hidden")
        ],
        className="nav",
    )


app.layout = html.Div([navigation(), html.Main(dash.page_container, className="page")])
