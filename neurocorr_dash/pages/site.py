import dash
from dash import Input, Output, callback, dcc, html

from neurocorr_dash import data, figures

dash.register_page(__name__, path="/site", name="Site", order=2)

EXAMPLE_SITE = 3453748112031  # the paper's example site (Fig 3)


def layout(scan=None, **_query):
    try:
        sites = data.sites()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    analyzed = sites[sites["analyzed"] & sites["mean_corr"].notna()]
    options = [
        {"label": f"site {row.site} ({row.ncells} cells)", "value": int(row.aod_scan_start_time)}
        for row in analyzed.itertuples()
    ]
    values = {o["value"] for o in options}
    value = (
        int(scan)
        if scan and int(scan) in values
        else (EXAMPLE_SITE if EXAMPLE_SITE in values else next(iter(values), None))
    )
    return html.Div(
        [
            html.H1("Site"),
            html.Div(
                [
                    dcc.Dropdown(options, value, id="site-choice", clearable=False, className="choice"),
                    dcc.RadioItems(
                        [{"label": data.METHOD_NAMES[m], "value": m} for m in (0, 10)],
                        0,
                        id="method-choice",
                        inline=True,
                        className="choice",
                    ),
                ],
                className="controls",
            ),
            html.Div([dcc.Graph(id="site-correlations"), dcc.Graph(id="site-cells")], className="row"),
        ]
    )


@callback(
    Output("site-correlations", "figure"),
    Output("site-cells", "figure"),
    Input("site-choice", "value"),
    Input("method-choice", "value"),
)
def show_site(aod_scan_start_time, method):
    label = data.site_label(aod_scan_start_time)
    try:
        R, xyz = data.correlation_matrix(aod_scan_start_time, method)
    except Exception:
        R, xyz = data.correlation_matrix(aod_scan_start_time, 0)
        method = 0
    return (
        figures.correlation_heatmap(R, f"Site {label}: {data.METHOD_NAMES[method]} correlations"),
        figures.cell_positions(xyz, f"Site {label}: {len(xyz)} active cells"),
    )
