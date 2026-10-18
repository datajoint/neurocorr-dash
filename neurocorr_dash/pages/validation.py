import dash
from dash import Input, Output, callback, dcc, html

from neurocorr_dash import data, figures

dash.register_page(__name__, path="/validation", name="Validation", order=3)


def layout(**_query):
    try:
        losses = data.validation_losses()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    methods = sorted(losses["method"].unique()) if not losses.empty else []
    reference = max(methods) if methods else None
    return html.Div(
        [
            html.H1("Cross-validated comparison"),
            html.P(
                "Each estimator is fitted on 9 of 10 folds of trials and evaluated on the held-out fold "
                "with the normal loss. Points are sites; the loss is averaged over the 10 folds. "
                "Values above zero favor the reference."
            ),
            html.Div(
                [
                    html.Label("Reference", htmlFor="reference-choice"),
                    dcc.Dropdown(
                        [{"label": data.METHOD_NAMES.get(m, str(m)), "value": int(m)} for m in methods],
                        reference,
                        id="reference-choice",
                        clearable=False,
                        className="choice",
                    ),
                ],
                className="controls",
            ),
            dcc.Graph(id="loss-differences"),
            html.P(id="loss-sites", className="note"),
        ]
    )


@callback(
    Output("loss-differences", "figure"), Output("loss-sites", "children"), Input("reference-choice", "value")
)
def show_losses(reference):
    losses = data.validation_losses()
    counts = losses.groupby("method").size()
    note = ", ".join(f"{data.METHOD_NAMES.get(m, m)}: {n} sites" for m, n in counts.items())
    return figures.loss_differences(losses, reference), f"Sites with all 10 folds — {note}." if note else ""
