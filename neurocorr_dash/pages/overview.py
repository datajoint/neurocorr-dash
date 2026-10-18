import dash
from dash import html

from neurocorr_dash import data

dash.register_page(__name__, path="/", name="Overview", order=0)

CITATION = (
    "Yatsenko D, Josić K, Ecker AS, Froudarakis E, Cotton RJ, Tolias AS (2015). "
    "Improved estimation and interpretation of correlations in neural circuits. "
    "PLOS Comput Biol 11(3): e1004083."
)


def table(df):
    return html.Table(
        [html.Thead(html.Tr([html.Th(c) for c in df.columns]))]
        + [html.Tbody([html.Tr([html.Td(v) for v in row]) for row in df.itertuples(index=False)])],
        className="table",
    )


def layout(**_query):
    try:
        sites = data.sites()
        progress = data.progress()
    except Exception as error:  # the pipeline may not be loaded yet
        return html.Div([html.H1("neurocorr"), html.P(f"The pipeline is not available: {error}")])
    analyzed = sites[sites["analyzed"]]
    return html.Div(
        [
            html.H1("Noise correlations in mouse visual cortex"),
            html.P(
                [
                    "A reproduction of ",
                    html.A(CITATION, href="https://doi.org/10.1371/journal.pcbi.1004083"),
                    " Original code and data: ",
                    html.A("atlab/cov-est", href="https://github.com/atlab/cov-est"),
                    " (MIT License, © 2014 Tolias Lab).",
                ],
                className="lede",
            ),
            html.Div(
                [
                    stat(len(sites), "correlation sets"),
                    stat(len(analyzed), "sites with more than 100 active cells"),
                    stat(int(analyzed["ncells"].sum()), "active cells at those sites"),
                ],
                className="stats",
            ),
            html.H2("Estimates computed"),
            html.P(
                "Each analyzed site gets 11 estimates per method: "
                "10 cross-validation folds and one from all trials."
            ),
            table(progress.rename(columns=str.capitalize)),
        ]
    )


def stat(value, label):
    return html.Div(
        [html.Div(f"{value:,}", className="stat-value"), html.Div(label, className="stat-label")],
        className="stat",
    )
