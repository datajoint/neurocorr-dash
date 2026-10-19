"""
Figures, built from plain arrays and data frames so they can be tested without a database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from neurocorr_dash.data import METHOD_NAMES

ACCENT = "#2563eb"
MUTED = "#94a3b8"
LAYOUT = {
    "template": "plotly_white",
    "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
    "font": {"family": "system-ui, -apple-system, Segoe UI, sans-serif", "size": 13},
}


def sites_scatter(sites: pd.DataFrame) -> go.Figure:
    """Mean sample correlation against the number of active cells, one point per site."""
    analyzed = sites[sites["analyzed"] & sites["mean_corr"].notna()]
    fig = go.Figure(
        go.Scatter(
            x=analyzed["ncells"],
            y=analyzed["mean_corr"],
            mode="markers",
            marker={"size": 9, "color": ACCENT},
            text=[f"site {s}" for s in analyzed["name"]],
            hovertemplate="%{text}<br>%{x} cells<br>mean correlation %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **LAYOUT,
        title=f"Mean sample correlation at {len(analyzed)} sites",
        xaxis_title="active cells",
        yaxis_title="mean sample correlation",
    )
    return fig


def correlation_heatmap(R: np.ndarray, title: str) -> go.Figure:
    """A correlation matrix with its diagonal blanked, on a symmetric color scale."""
    off = np.array(R, dtype=float)
    np.fill_diagonal(off, np.nan)
    limit = float(np.nanpercentile(np.abs(off), 99)) or 0.1
    fig = go.Figure(
        go.Heatmap(
            z=off,
            colorscale="RdBu",
            reversescale=True,
            zmin=-limit,
            zmax=limit,
            colorbar={"title": "r"},
            hovertemplate="cells %{y}, %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**LAYOUT, title=title, height=520, width=560)
    fig.update_yaxes(autorange="reversed", scaleanchor="x", title="cell")
    fig.update_xaxes(title="cell")
    return fig


def cell_positions(xyz: np.ndarray, title: str) -> go.Figure:
    """Cell centers seen from above, colored by depth."""
    fig = go.Figure(
        go.Scatter(
            x=xyz[:, 0],
            y=xyz[:, 1],
            mode="markers",
            marker={"size": 7, "color": xyz[:, 2], "colorscale": "Viridis", "colorbar": {"title": "z (µm)"}},
            hovertemplate="x %{x:.0f} µm, y %{y:.0f} µm<extra></extra>",
        )
    )
    fig.update_layout(**LAYOUT, title=title, height=520, width=560)
    fig.update_xaxes(title="x (µm)")
    fig.update_yaxes(title="y (µm)", scaleanchor="x")
    return fig


def loss_differences(losses: pd.DataFrame, reference: int) -> go.Figure:
    """
    Validation loss of each method minus the reference method's, per site.

    Positive values mean the reference does better. Only sites where both methods have
    all 10 folds are compared, as in the paper's Fig 4.
    """
    table = losses.pivot(index="site", columns="method", values="cv_loss")
    fig = go.Figure()
    if reference in table:
        for method in [m for m in table.columns if m != reference]:
            difference = (table[method] - table[reference]).dropna()
            if difference.empty:
                continue
            fig.add_trace(
                go.Box(
                    y=difference,
                    name=METHOD_NAMES.get(method, str(method)),
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    marker={"color": ACCENT, "size": 6},
                    line={"color": MUTED},
                    text=[f"site {s}" for s in difference.index],
                    hovertemplate="%{text}<br>%{y:+.4f}<extra></extra>",
                )
            )
    if not fig.data:
        fig.add_annotation(
            text="No other method has all 10 folds at a site yet",
            showarrow=False,
            font={"color": MUTED, "size": 15},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    fig.add_hline(y=0, line={"color": MUTED, "dash": "dot"})
    fig.update_layout(
        **LAYOUT,
        title=f"Validation loss relative to {METHOD_NAMES.get(reference, reference)}",
        yaxis_title="Δ loss (nats/cell/bin)",
        showlegend=False,
    )
    return fig


def fig3e(R: np.ndarray, label: str) -> go.Figure:
    """Fig 3 E: the sample noise-correlation matrix of a site, on a fixed ±0.25 scale."""
    off = np.array(R, dtype=float)
    np.fill_diagonal(off, np.nan)
    n = len(off)
    fig = go.Figure(
        go.Heatmap(
            z=off,
            x=np.arange(1, n + 1),
            y=np.arange(1, n + 1),
            colorscale="RdBu",
            reversescale=True,
            zmin=-0.25,
            zmax=0.25,
            colorbar={"title": "r", "tickvals": [-0.25, 0, 0.25]},
            hovertemplate="cells %{y} and %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**LAYOUT, title=f"E. Sample correlations, {label} ({n} cells)", height=520, width=560)
    fig.update_yaxes(autorange="reversed", scaleanchor="x", title="cell number")
    fig.update_xaxes(title="cell number")
    return fig


def fig3f(R: np.ndarray, label: str) -> go.Figure:
    """Fig 3 F: the distribution of a site's pairwise sample correlations, with its mean."""
    values = np.asarray(R)[np.triu_indices_from(np.asarray(R), 1)]
    mean = float(values.mean())
    fig = go.Figure(
        go.Histogram(
            x=values,
            xbins={"start": -0.15, "end": 0.3, "size": 0.002},
            marker={"color": "#9ea3c4", "line": {"color": "#1f2a5c", "width": 0.3}},
            hovertemplate="r %{x}<br>%{y} pairs<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": "black", "width": 1})
    fig.add_vline(x=mean, line={"color": "red", "width": 3})
    fig.add_annotation(
        x=mean,
        y=1,
        yref="paper",
        text=f"mean {mean:.4f}",
        showarrow=False,
        xanchor="left",
        font={"color": "red"},
    )
    fig.update_layout(
        **LAYOUT,
        title=f"F. Distribution of {len(values):,} correlations",
        height=520,
        width=560,
        bargap=0,
        showlegend=False,
    )
    fig.update_xaxes(title="correlations", range=[-0.1, 0.21])
    fig.update_yaxes(title="cell pairs")
    return fig
