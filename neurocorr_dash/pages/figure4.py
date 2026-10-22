import dash
from dash import html, dcc
import pandas as pd
import plotly.graph_objects as go

from neurocorr_dash import data
from neurocorr_dash.figures import LAYOUT, ACCENT, MUTED

dash.register_page(__name__, path="/figure4", name="Figure 4", order=4.4)


def layout(**_query):
    try:
        losses = data.validation_losses()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    
    if losses.empty:
        return html.Div([
            html.H1("Figure 4: Method Comparison"),
            html.P("No validation data available yet.")
        ])
    
    # Filter to sites with all 5 required methods
    required_methods = [0, 10, 30, 80, 90]
    table = losses.pivot(index="site", columns="method", values="cv_loss")
    complete_sites = table.dropna(subset=required_methods)
    
    if complete_sites.empty:
        return html.Div([
            html.H1("Figure 4: Method Comparison"),
            html.P("Waiting for all methods to complete across sites with 10 folds.")
        ])
    
    n_sites = len(complete_sites)
    
    # Compute differences relative to method 90
    differences = {}
    medians = {}
    for method in [0, 10, 30, 80]:
        diff = complete_sites[method] - complete_sites[90]
        differences[method] = diff
        medians[method] = float(diff.median())
    
    # Create horizontal box plots
    fig = go.Figure()
    
    methods_ordered = [80, 30, 10, 0]
    labels = {
        0: "sample (0)",
        10: "diagonal shrinkage (10)",
        30: "factor analysis (30)",
        80: "sparse (80)"
    }
    
    for method in methods_ordered:
        diff = differences[method]
        fig.add_trace(
            go.Box(
                x=diff,
                name=labels[method],
                orientation='h',
                boxmean=False,
                marker={"color": ACCENT},
                line={"color": MUTED},
                hovertemplate=f"{labels[method]}<br>%{{x:+.4f}} nats/cell/bin<extra></extra>"
            )
        )
    
    # Add dotted line at zero
    fig.add_vline(x=0, line={"color": "black", "dash": "dot", "width": 1.5})
    
    fig.update_layout(
        **LAYOUT,
        title=f"Validation loss relative to sparse+latent (90) across {n_sites} sites",
        xaxis_title="Validation loss difference (nats per cell per bin)",
        yaxis_title="",
        showlegend=False,
        height=400
    )
    
    # Format medians for display
    median_text = html.Div([
        html.H3("Median excess loss (nats per cell per bin)"),
        html.Ul([
            html.Li(f"{labels[0]}: {medians[0]:.4f}"),
            html.Li(f"{labels[10]}: {medians[10]:.4f}"),
            html.Li(f"{labels[30]}: {medians[30]:.4f}"),
            html.Li(f"{labels[80]}: {medians[80]:.4f}")
        ])
    ], className="medians")
    
    return html.Div([
        html.H1("Figure 4: Method Comparison"),
        html.P(
            f"At each of {n_sites} analyzed sites, the 10-fold validation loss of sample (0), "
            "diagonal shrinkage (10), factor analysis (30), and sparse (80) estimates "
            "compared to sparse+latent (90), averaged over folds. "
            "Positive values indicate worse performance than sparse+latent."
        ),
        dcc.Graph(figure=fig),
        median_text
    ])
