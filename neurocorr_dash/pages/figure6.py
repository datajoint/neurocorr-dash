import dash
from dash import html, dcc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neurocorr_dash import data
from neurocorr_dash.figures import LAYOUT, ACCENT

dash.register_page(__name__, path="/figure6", name="Figure 6", order=6)


def layout(**_query):
    try:
        c = data.covest()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    
    # Fetch method 100, nfolds 1 data (the analyzed sites)
    method100 = (c.CovMatrix & 'method = 100' & 'nfolds = 1').to_pandas().reset_index()
    
    if method100.empty:
        return html.Div([
            html.H1("Figure 6: Site Analysis"),
            html.P("No method 100 (sparse+latent, all-trial) estimates available yet.")
        ])
    
    # Fetch method 0 data (sample correlation)
    method0 = (c.CovMatrix & 'method = 0' & 'nfolds = 1').to_pandas().reset_index()
    
    # Get trace set info
    trace_info = c.TrialTraceSet.proj('session', 'site').to_pandas().reset_index()
    
    # Merge method0 and trace_info
    method0 = method0.merge(trace_info, on='trace_set', how='left')
    method100 = method100.merge(trace_info, on='trace_set', how='left')
    
    # Build dataset
    results = []
    for _, m100 in method100.iterrows():
        ts = m100['trace_set']
        m0_row = method0[method0['trace_set'] == ts]
        if m0_row.empty:
            continue
        
        # Fetch arrays
        corr100 = np.asarray(m100['corr_matrix'])
        corr0 = np.asarray(m0_row.iloc[0]['corr_matrix'])
        sparse = np.asarray(m100['sparse'])
        lowrank = np.asarray(m100['lowrank'])
        
        ncells = corr100.shape[0]
        nlatent = lowrank.shape[1]
        
        # Percent connectivity
        n = sparse.shape[0]
        off_diag_mask = ~np.eye(n, dtype=bool)
        nonzero_off_diag = np.count_nonzero(sparse[off_diag_mask])
        total_off_diag = n * (n - 1)
        pct_connectivity = 100 * nonzero_off_diag / total_off_diag
        
        # Average sample correlation
        avg_sample_corr = float(corr0[off_diag_mask].mean())
        
        # Average partial correlation
        precision100 = np.linalg.inv(corr100)
        avg_partial_corr = float(-precision100[off_diag_mask].mean())
        
        # Percent negative interactions
        positive_sparse_off_diag = np.sum(sparse[off_diag_mask] > 0)
        pct_negative_interactions = (
            100 * positive_sparse_off_diag / nonzero_off_diag if nonzero_off_diag > 0 else 0
        )
        
        results.append({
            'trace_set': int(ts),
            'session': int(m100['session']),
            'site': int(m100['site']),
            'ncells': int(ncells),
            'nlatent': int(nlatent),
            'pct_connectivity': float(pct_connectivity),
            'avg_sample_corr': float(avg_sample_corr),
            'avg_partial_corr': float(avg_partial_corr),
            'pct_negative_interactions': float(pct_negative_interactions)
        })
    
    if not results:
        return html.Div([
            html.H1("Figure 6: Site Analysis"),
            html.P("Waiting for method 0 and method 100 estimates to complete.")
        ])
    
    # Sort by ncells for consistent coloring
    results = sorted(results, key=lambda x: x['ncells'])
    
    # Extract arrays
    trace_sets = np.array([r['trace_set'] for r in results])
    ncells = np.array([r['ncells'] for r in results])
    nlatent = np.array([r['nlatent'] for r in results])
    pct_connectivity = np.array([r['pct_connectivity'] for r in results])
    avg_sample_corr = np.array([r['avg_sample_corr'] for r in results])
    avg_partial_corr = np.array([r['avg_partial_corr'] for r in results])
    pct_negative_interactions = np.array([r['pct_negative_interactions'] for r in results])
    
    # Find target (trace set 28, session 8828, site 2031)
    target_mask = trace_sets == 28
    has_target = target_mask.any()
    
    # Calculate CV
    cv_sample = float(np.std(avg_sample_corr) / np.mean(avg_sample_corr))
    cv_partial = float(np.std(avg_partial_corr) / np.mean(avg_partial_corr))
    
    # Create hover text
    hover_text = [
        f"trace set {ts}<br>session {r['session']}, site {r['site']}<br>{r['ncells']} cells"
        for ts, r in zip(trace_sets, results)
    ]
    
    # Create 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("A. Latent units vs cells", "B. Connectivity vs cells",
                       "C. Partial vs sample correlation", "D. Negative interactions vs connectivity"),
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )
    
    # Panel A: Number of latent units vs number of cells
    fig.add_trace(
        go.Scatter(
            x=ncells, y=nlatent,
            mode='markers',
            marker=dict(size=10, color=ncells, colorscale='Viridis', showscale=False,
                       line=dict(width=0.5, color='black')),
            text=hover_text,
            hovertemplate='%{text}<br>%{y} latent units<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    if has_target:
        target_idx = np.where(target_mask)[0][0]
        fig.add_trace(
            go.Scatter(
                x=[ncells[target_idx]], y=[nlatent[target_idx]],
                mode='markers',
                marker=dict(size=20, color='rgba(0,0,0,0)', line=dict(width=3, color='red')),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Panel B: Percent connectivity vs number of cells
    fig.add_trace(
        go.Scatter(
            x=ncells, y=pct_connectivity,
            mode='markers',
            marker=dict(size=10, color=ncells, colorscale='Viridis', showscale=False,
                       line=dict(width=0.5, color='black')),
            text=hover_text,
            hovertemplate='%{text}<br>%{y:.2f}% connectivity<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    if has_target:
        fig.add_trace(
            go.Scatter(
                x=[ncells[target_idx]], y=[pct_connectivity[target_idx]],
                mode='markers',
                marker=dict(size=20, color='rgba(0,0,0,0)', line=dict(width=3, color='red')),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Panel C: Average partial vs sample correlation
    fig.add_trace(
        go.Scatter(
            x=avg_sample_corr, y=avg_partial_corr,
            mode='markers',
            marker=dict(size=10, color=ncells, colorscale='Viridis', showscale=True,
                       line=dict(width=0.5, color='black'),
                       colorbar=dict(title="cells", x=1.15, len=0.45, y=0.73)),
            text=hover_text,
            hovertemplate='%{text}<br>sample: %{x:.4f}<br>partial: %{y:.4f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    if has_target:
        fig.add_trace(
            go.Scatter(
                x=[avg_sample_corr[target_idx]], y=[avg_partial_corr[target_idx]],
                mode='markers',
                marker=dict(size=20, color='rgba(0,0,0,0)', line=dict(width=3, color='red')),
                hoverinfo='skip',
                showlegend=False
            ),
            row=2, col=1
        )
    # Add diagonal
    corr_min = min(avg_sample_corr.min(), avg_partial_corr.min())
    corr_max = max(avg_sample_corr.max(), avg_partial_corr.max())
    fig.add_trace(
        go.Scatter(
            x=[corr_min, corr_max], y=[corr_min, corr_max],
            mode='lines',
            line=dict(color='black', dash='dash', width=1),
            hoverinfo='skip',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Panel D: Percent negative interactions vs percent connectivity
    fig.add_trace(
        go.Scatter(
            x=pct_connectivity, y=pct_negative_interactions,
            mode='markers',
            marker=dict(size=10, color=ncells, colorscale='Viridis', showscale=False,
                       line=dict(width=0.5, color='black')),
            text=hover_text,
            hovertemplate='%{text}<br>connectivity: %{x:.2f}%<br>negative: %{y:.2f}%<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    if has_target:
        fig.add_trace(
            go.Scatter(
                x=[pct_connectivity[target_idx]], y=[pct_negative_interactions[target_idx]],
                mode='markers',
                marker=dict(size=20, color='rgba(0,0,0,0)', line=dict(width=3, color='red')),
                hoverinfo='skip',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update axes labels
    fig.update_xaxes(title_text="Number of cells", row=1, col=1)
    fig.update_yaxes(title_text="Latent units", row=1, col=1)
    
    fig.update_xaxes(title_text="Number of cells", row=1, col=2)
    fig.update_yaxes(title_text="Connectivity (%)", row=1, col=2)
    
    fig.update_xaxes(title_text="Average sample correlation", row=2, col=1)
    fig.update_yaxes(title_text="Average partial correlation", row=2, col=1)
    
    fig.update_xaxes(title_text="Connectivity (%)", row=2, col=2)
    fig.update_yaxes(title_text="Negative interactions (%)", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        **LAYOUT,
        height=800,
        title_text=f"Sparse+latent estimates across {len(results)} sites (method 100, all trials)",
        showlegend=False
    )
    
    # Format statistics
    stats_text = html.Div([
        html.H3("Coefficient of variation across sites"),
        html.Ul([
            html.Li(f"Average sample correlation: {cv_sample:.3f}"),
            html.Li(f"Average partial correlation: {cv_partial:.3f}")
        ]),
        html.P(
            f"Red circle marks trace set 28 (session 8828, site 2031)." if has_target else 
            "Target site (trace set 28, session 8828, site 2031) not found in dataset.",
            style={'marginTop': '1em', 'fontStyle': 'italic'}
        )
    ], className="stats")
    
    return html.Div([
        html.H1("Figure 6: Analysis Across Sites"),
        html.P(
            f"Metrics from sparse+latent (method 100) estimates on all trials across {len(results)} analyzed sites. "
            "A: Number of latent units scales with cells. "
            "B: Sparse component connectivity varies widely. "
            "C: Partial correlations are consistently smaller than sample correlations. "
            "D: More connected sites have higher rates of negative interactions (positive sparse component entries)."
        ),
        dcc.Graph(figure=fig),
        stats_text
    ])
