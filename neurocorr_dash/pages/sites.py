import dash
from dash import dcc, html

from neurocorr_dash import data, figures

dash.register_page(__name__, path="/sites", name="Sites", order=1)

COLUMNS = {
    "site": "Site",
    "ndirs": "Directions",
    "nneurons": "Cells",
    "ncells": "Active cells",
    "mean_corr": "Mean sample correlation",
}


def layout():
    try:
        sites = data.sites()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    rows = sites[list(COLUMNS)].rename(columns=COLUMNS)
    rows["Mean sample correlation"] = rows["Mean sample correlation"].map(
        lambda v: "" if v != v else f"{v:.4f}"
    )
    return html.Div(
        [
            html.H1("Sites"),
            html.P(
                "Each site is a population in layers 2/3 of mouse V1, recorded with 3D random-access "
                "two-photon microscopy. Sites with more than 100 active cells are analyzed."
            ),
            dcc.Graph(figure=figures.sites_scatter(sites)),
            html.Table(
                [html.Thead(html.Tr([html.Th(c) for c in rows.columns]))]
                + [
                    html.Tbody(
                        [
                            html.Tr([html.Td(v) for v in row], className="" if ok else "excluded")
                            for row, ok in zip(rows.itertuples(index=False), sites["analyzed"])
                        ]
                    )
                ],
                className="table",
            ),
        ]
    )
