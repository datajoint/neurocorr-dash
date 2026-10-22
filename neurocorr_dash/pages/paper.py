import dash
from dash import Input, Output, callback, dcc, html

from neurocorr_dash import data, figures

dash.register_page(__name__, path="/figures", name="Figures", order=4)

EXAMPLE_SITE = 2031  # the example site of the paper's Fig 3, numbered as the original code numbers it


def layout(**_query):
    try:
        sites = data.sites()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    ready = sites[sites["analyzed"] & sites["mean_corr"].notna()]
    options = [
        {"label": f"{data.site_label(row)}, {row.ncells} cells", "value": int(row.trace_set)}
        for row in ready.itertuples()
    ]
    example = ready.loc[ready["site"] == EXAMPLE_SITE, "trace_set"]
    value = int(example.iloc[0]) if len(example) else next((o["value"] for o in options), None)
    return html.Div(
        [
            html.H1("Figure 3"),
            html.P(
                "The figures of Yatsenko et al. (2015), drawn from the pipeline's results. "
                "Fig 3 E and F: the sample noise correlations of one site (method 0, all trials) "
                "and their distribution. The paper's example site (292 cells, mean 0.020) is site 2031, "
                "as the original code numbers it, in session 8828.",
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
def show_fig3(trace_set):
    R, _ = data.correlation_matrix(trace_set, 0)
    sites = data.sites()
    site = data.site_label(sites[sites["trace_set"] == trace_set].iloc[0])
    return figures.fig3e(R, site), figures.fig3f(R, site)
