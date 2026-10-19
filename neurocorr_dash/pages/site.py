import dash
from dash import Input, Output, callback, dcc, html

from neurocorr_dash import data, figures

dash.register_page(__name__, path="/site", name="Site", order=2)

EXAMPLE_SITE = 2031  # the paper's example site (Fig 3), by its number in the paper


def layout(site=None, **_query):
    try:
        sites = data.sites()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    analyzed = sites[sites["analyzed"] & sites["mean_corr"].notna()]
    options = [
        {"label": f"{data.site_label(row)}, {row.ncells} cells", "value": int(row.trace_set)}
        for row in analyzed.itertuples()
    ]
    # ?site=8-4: session 8, site 4
    by_name = dict(zip(analyzed["name"], analyzed["trace_set"]))
    example = analyzed.loc[analyzed["paper_site"] == EXAMPLE_SITE, "trace_set"]
    value = (
        int(by_name[site])
        if site in by_name
        else (int(example.iloc[0]) if len(example) else next(iter(by_name.values()), None))
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
def show_site(trace_set, method):
    sites = data.sites()
    label = data.site_label(sites[sites["trace_set"] == trace_set].iloc[0])
    try:
        R, xyz = data.correlation_matrix(trace_set, method)
    except Exception:
        R, xyz = data.correlation_matrix(trace_set, 0)
        method = 0
    return (
        figures.correlation_heatmap(
            R, f"{label[0].upper()}{label[1:]}: {data.METHOD_NAMES[method]} correlations"
        ),
        figures.cell_positions(xyz, f"{label[0].upper()}{label[1:]}: {len(xyz)} active cells"),
    )
