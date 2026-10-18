import dash
from dash import Input, Output, callback, dcc, html

from neurocorr_dash import data, figures

dash.register_page(__name__, path="/figures", name="Figures", order=4)

EXAMPLE_SITE = 3453748112031  # the paper's example site (Fig 3)


def layout(**_query):
    try:
        sites = data.sites()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    ready = sites[sites["analyzed"] & sites["mean_corr"].notna()]
    options = [
        {"label": f"site {row.site} ({row.ncells} cells)", "value": int(row.aod_scan_start_time)}
        for row in ready.itertuples()
    ]
    values = {o["value"] for o in options}
    value = EXAMPLE_SITE if EXAMPLE_SITE in values else next(iter(values), None)
    return html.Div(
        [
            html.H1("Figures"),
            html.P(
                "The figures of Yatsenko et al. (2015), drawn from the pipeline's results. "
                "Fig 3 E and F: the sample noise correlations of one site (method 0, all trials) "
                "and their distribution. The paper's example site is 2031 (292 cells, mean 0.020).",
                className="lede",
            ),
            html.Div(
                [dcc.Dropdown(options, value, id="figure-site", clearable=False, className="choice")],
                className="controls",
            ),
            html.Div([dcc.Graph(id="fig3e"), dcc.Graph(id="fig3f")], className="row"),
        ]
    )


@callback(Output("fig3e", "figure"), Output("fig3f", "figure"), Input("figure-site", "value"))
def show_fig3(aod_scan_start_time):
    R, _ = data.correlation_matrix(aod_scan_start_time, 0)
    site = data.site_label(aod_scan_start_time)
    return figures.fig3e(R, site), figures.fig3f(R, site)
