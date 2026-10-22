import dash
from dash import html, dcc, callback, Input, Output
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neurocorr_dash import data
from neurocorr_dash.figures import LAYOUT, ACCENT, MUTED

dash.register_page(__name__, path="/figure5", name="Figure 5", order=5)


def layout(**_query):
    try:
        c = data.covest()
        # Find sites with both method 0 and method 100 computed
        method0_sites = (c.CovMatrix & "method = 0" & "k = 1" & "nfolds = 1").proj("trace_set").to_pandas()
        method100_sites = (c.CovMatrix & "method = 100" & "k = 1" & "nfolds = 1").proj("trace_set").to_pandas()
        available = set(method0_sites["trace_set"]) & set(method100_sites["trace_set"])
        
        if not available:
            return html.Div([
                html.H1("Figure 5: Correlation Matrix Decomposition"),
                html.P("No sites with both sample and sparse+latent estimates available yet.")
            ])
        
        # Get site details
        site_info = (c.TrialTraceSet * c.Site & [{"trace_set": ts} for ts in available]).proj(
            "session", "site", "nneurons"
        ).to_pandas().reset_index()
        
        site_options = [
            {"label": f"Session {row['session']}, Site {row['site']} ({row['nneurons']} cells)",
             "value": int(row["trace_set"])}
            for _, row in site_info.sort_values(["session", "site"]).iterrows()
        ]
        
        # Default to session 8828, site 2031 if available, otherwise first site
        default_site = 28 if 28 in available else int(site_info.iloc[0]["trace_set"])
        
        return html.Div([
            html.H1("Figure 5: Correlation Matrix Decomposition"),
            html.P([
                "Comparison of sample correlations (method 0) and sparse+latent decomposition (method 100). ",
                "The sparse+latent method decomposes the precision matrix as S - LL", html.Sup("T"), 
                ", where S is sparse and L is low-rank."
            ]),
            html.Div([
                html.Label("Select site:"),
                dcc.Dropdown(
                    id="figure5-site-dropdown",
                    options=site_options,
                    value=default_site,
                    clearable=False,
                    style={"width": "400px"}
                )
            ], style={"marginBottom": "20px"}),
            html.Div(id="figure5-stats", style={"marginBottom": "20px"}),
            dcc.Graph(id="figure5-plot", style={"height": "900px"})
        ])
    
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")


@callback(
    [Output("figure5-plot", "figure"),
     Output("figure5-stats", "children")],
    Input("figure5-site-dropdown", "value")
)
def update_figure5(trace_set):
    c = data.covest()
    
    # Fetch data
    c0_data = (c.CovMatrix & {"trace_set": trace_set, "method": 0, "k": 1, "nfolds": 1}).fetch1()
    c1_data = (c.CovMatrix & {"trace_set": trace_set, "method": 100, "k": 1, "nfolds": 1}).fetch1()
    site_data = (c.TrialTraceSet * c.Site & {"trace_set": trace_set}).fetch1()
    
    C0 = np.asarray(c0_data["corr_matrix"])
    C1 = np.asarray(c1_data["corr_matrix"])
    S = np.asarray(c1_data["sparse"])
    L = np.asarray(c1_data["lowrank"])
    sparsity_val = float(c1_data["sparsity"])
    
    ncells = C0.shape[0]
    session = int(site_data["session"])
    site = int(site_data["site"])
    
    # Compute precision matrices
    P0 = np.linalg.inv(C0)
    P1 = np.linalg.inv(C1)
    
    # Compute partial correlations (normalize by diagonal)
    PC0 = -P0 / np.sqrt(np.outer(np.diag(P0), np.diag(P0)))
    np.fill_diagonal(PC0, 1.0)
    
    PC1 = -P1 / np.sqrt(np.outer(np.diag(P1), np.diag(P1)))
    np.fill_diagonal(PC1, 1.0)
    
    PCS = -S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    np.fill_diagonal(PCS, 1.0)
    
    LLT = L @ L.T
    PLLT = LLT / np.sqrt(np.outer(np.diag(LLT), np.diag(LLT)))
    np.fill_diagonal(PLLT, 1.0)
    
    # Calculate statistics
    n_latent = L.shape[1]
    off_diag_mask = ~np.eye(ncells, dtype=bool)
    n_nonzero = int(np.sum((S[off_diag_mask] != 0)))
    n_possible = ncells * (ncells - 1)
    avg_interactions = n_nonzero / ncells
    
    # Helper function to create split matrix
    def split_matrix(below, above):
        split = np.zeros_like(below)
        tril_idx = np.tril_indices(ncells, k=-1)
        triu_idx = np.triu_indices(ncells, k=1)
        split[tril_idx] = below[tril_idx]
        split[triu_idx] = above[triu_idx]
        np.fill_diagonal(split, 1.0)
        return split
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "A: Correlations<br>(below: sample, above: sparse+latent)",
            "B: Correlation comparison",
            "C: Partial correlations<br>(below: sample, above: regularized)",
            "D: Partial correlation comparison",
            "E: Decomposition<br>(below: sparse, above: LL<sup>T</sup>)",
            "F: Sparse vs sample comparison"
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )
    
    # A: Sample correlations (below) vs C1 correlations (above)
    split_corr = split_matrix(C0, C1)
    fig.add_trace(
        go.Heatmap(
            z=split_corr,
            colorscale="RdBu",
            reversescale=True,
            zmin=-0.1,
            zmax=0.1,
            colorbar=dict(x=0.31, len=0.4, y=0.75),
            hovertemplate="Cell %{y}, %{x}<br>r = %{z:.4f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # B: 2D histogram of C1 vs sample correlations
    triu_idx = np.triu_indices(ncells, k=1)
    c0_upper = C0[triu_idx]
    c1_upper = C1[triu_idx]
    
    fig.add_trace(
        go.Histogram2d(
            x=c0_upper,
            y=c1_upper,
            nbinsx=50,
            nbinsy=50,
            colorscale="Viridis",
            colorbar=dict(x=0.645, len=0.4, y=0.75),
            hovertemplate="Sample: %{x:.4f}<br>Sparse+latent: %{y:.4f}<br>Count: %{z}<extra></extra>"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[-0.1, 0.1],
            y=[-0.1, 0.1],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=2
    )
    
    # C: Sample partial correlations (below) vs regularized partial correlations (above)
    split_pc = split_matrix(PC0, PC1)
    fig.add_trace(
        go.Heatmap(
            z=split_pc,
            colorscale="RdBu",
            reversescale=True,
            zmin=-0.1,
            zmax=0.1,
            colorbar=dict(x=0.98, len=0.4, y=0.75),
            hovertemplate="Cell %{y}, %{x}<br>partial r = %{z:.4f}<extra></extra>"
        ),
        row=1, col=3
    )
    
    # D: 2D histogram of regularized vs sample partial correlations
    pc0_upper = PC0[triu_idx]
    pc1_upper = PC1[triu_idx]
    
    fig.add_trace(
        go.Histogram2d(
            x=pc0_upper,
            y=pc1_upper,
            nbinsx=50,
            nbinsy=50,
            colorscale="Viridis",
            colorbar=dict(x=0.31, len=0.4, y=0.25),
            hovertemplate="Sample: %{x:.4f}<br>Regularized: %{y:.4f}<br>Count: %{z}<extra></extra>"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[-0.1, 0.1],
            y=[-0.1, 0.1],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=2, col=1
    )
    
    # E: Sparse partial correlations (below) vs low-rank component (above)
    split_decomp = split_matrix(PCS, PLLT)
    fig.add_trace(
        go.Heatmap(
            z=split_decomp,
            colorscale="RdBu",
            reversescale=True,
            zmin=-0.1,
            zmax=0.1,
            colorbar=dict(x=0.645, len=0.4, y=0.25),
            hovertemplate="Cell %{y}, %{x}<br>partial r = %{z:.4f}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # F: 2D histogram of sparse partial correlations vs sample correlations
    pcs_upper = PCS[triu_idx]
    
    fig.add_trace(
        go.Histogram2d(
            x=c0_upper,
            y=pcs_upper,
            nbinsx=50,
            nbinsy=50,
            colorscale="Viridis",
            colorbar=dict(x=0.98, len=0.4, y=0.25),
            hovertemplate="Sample r: %{x:.4f}<br>Sparse partial r: %{y:.4f}<br>Count: %{z}<extra></extra>"
        ),
        row=2, col=3
    )
    fig.add_trace(
        go.Scatter(
            x=[-0.1, 0.1],
            y=[-0.1, 0.1],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=2, col=3
    )
    
    # Update all x and y axes
    for i in range(1, 7):
        row = (i - 1) // 3 + 1
        col = (i - 1) % 3 + 1
        
        if col == 1:  # Heatmaps
            fig.update_xaxes(title_text="Cell", row=row, col=col)
            fig.update_yaxes(title_text="Cell", row=row, col=col, autorange="reversed")
        else:  # Scatter/histogram plots
            fig.update_xaxes(range=[-0.1, 0.1], row=row, col=col)
            fig.update_yaxes(range=[-0.1, 0.1], row=row, col=col)
    
    fig.update_layout(
        **LAYOUT,
        title=f"Session {session}, Site {site} (n={ncells} cells)",
        height=900,
        showlegend=False
    )
    
    # Statistics display
    stats = html.Div([
        html.H3("Decomposition Statistics"),
        html.Ul([
            html.Li(f"Number of cells: {ncells}"),
            html.Li(f"Sparsity of S: {sparsity_val:.1%} (share of zero off-diagonal entries)"),
            html.Li(f"Average interactions per cell: {avg_interactions:.1f} ({n_nonzero:,} of {n_possible:,} possible)"),
            html.Li(f"Number of latent units: {n_latent}")
        ])
    ], className="stats")
    
    return fig, stats
