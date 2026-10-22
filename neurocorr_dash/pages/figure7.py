import dash
from dash import html, dcc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neurocorr_dash import data
from neurocorr_dash.figures import LAYOUT

dash.register_page(__name__, path="/figure7", name="Figure 7", order=7)


def compute_pairwise_diffs(cell_xyz, von_pref, tuned, C0, partial_corr, pos_connected, neg_connected):
    """Compute pairwise differences for all cell pairs."""
    n = len(cell_xyz)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Position differences
    dx = np.abs(cell_xyz[i_idx, 0] - cell_xyz[j_idx, 0])
    dy = np.abs(cell_xyz[i_idx, 1] - cell_xyz[j_idx, 1])
    dz = np.abs(cell_xyz[i_idx, 2] - cell_xyz[j_idx, 2])
    d_lateral = np.sqrt(dx**2 + dy**2)
    
    # Tuned pairs
    tuned_pairs = tuned[i_idx] & tuned[j_idx]
    
    # Orientation differences (fold to 0-90°)
    ori_i = von_pref[i_idx] % np.pi
    ori_j = von_pref[j_idx] % np.pi
    d_ori = np.abs(ori_i - ori_j)
    d_ori = np.minimum(d_ori, np.pi - d_ori)
    d_ori_deg = np.degrees(d_ori)
    
    # Get values
    sample_corr = C0[i_idx, j_idx]
    partial_corr_vals = partial_corr[i_idx, j_idx]
    pos_conn = pos_connected[i_idx, j_idx]
    neg_conn = neg_connected[i_idx, j_idx]
    
    return {
        'd_lateral': d_lateral,
        'dz': dz,
        'd_ori_deg': d_ori_deg,
        'tuned_pairs': tuned_pairs,
        'sample_corr': sample_corr,
        'partial_corr': partial_corr_vals,
        'pos_conn': pos_conn,
        'neg_conn': neg_conn,
    }


def bin_pairs(pairs, bin_edges, key, mask=None):
    """Bin pairs and compute mean statistics."""
    if mask is None:
        mask = np.ones(len(pairs[key]), dtype=bool)
    
    values = pairs[key][mask]
    n_bins = len(bin_edges) - 1
    bin_means = {
        'sample_corr': [],
        'partial_corr': [],
        'frac_pos': [],
        'frac_neg': [],
        'n_pairs': [],
        'bin_centers': [],
    }
    
    for i in range(n_bins):
        bin_mask = (values >= bin_edges[i]) & (values < bin_edges[i+1])
        n = bin_mask.sum()
        
        if n >= 200:  # Minimum 200 pairs
            bin_means['n_pairs'].append(int(n))
            bin_means['bin_centers'].append(float((bin_edges[i] + bin_edges[i+1]) / 2))
            
            idx = np.where(mask)[0][bin_mask]
            bin_means['sample_corr'].append(float(pairs['sample_corr'][idx].mean()))
            bin_means['partial_corr'].append(float(pairs['partial_corr'][idx].mean()))
            bin_means['frac_pos'].append(float(pairs['pos_conn'][idx].mean()))
            bin_means['frac_neg'].append(float(pairs['neg_conn'][idx].mean()))
    
    return bin_means


def layout(**_query):
    try:
        c = data.covest()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    
    # Define the five high-connectivity sites
    sites_info = [
        {'trace_set': 30, 'ori_trace_set': 29, 'label': 'Site 30'},
        {'trace_set': 26, 'ori_trace_set': 25, 'label': 'Site 26'},
        {'trace_set': 28, 'ori_trace_set': 27, 'label': 'Site 28'},
        {'trace_set': 12, 'ori_trace_set': 11, 'label': 'Site 12'},
        {'trace_set': 45, 'ori_trace_set': 44, 'label': 'Site 45'},
    ]
    
    # Load data for all sites
    all_data = []
    for site_info in sites_info:
        ts = site_info['trace_set']
        ori_ts = site_info['ori_trace_set']
        
        try:
            # Fetch CovMatrix for methods 0 and 91
            cov_query = c.CovMatrix & f'trace_set = {ts}' & 'method IN (0, 91)' & 'k = 1' & 'nfolds = 1'
            cov_data = cov_query.to_pandas().reset_index()
            
            if len(cov_data) != 2:
                continue
            
            method_0 = cov_data[cov_data['method'] == 0].iloc[0]
            method_91 = cov_data[cov_data['method'] == 91].iloc[0]
            
            # Fetch cell selection
            cell_sel = np.asarray((c.CellSelection & f'trace_set = {ts}').to_arrays('selection')[0])
            ncells = int(cell_sel.sum())
            
            # Fetch cell positions
            cell_xyz = np.asarray((c.TrialTraceSet & f'trace_set = {ts}').to_arrays('cell_xyz')[0])
            cell_xyz = cell_xyz[cell_sel]
            
            # Fetch orientation tuning
            von_p = np.asarray((c.OriTuning & f'trace_set = {ori_ts}').to_arrays('von_p_value')[0])
            von_pref = np.asarray((c.OriTuning & f'trace_set = {ori_ts}').to_arrays('von_pref')[0])
            von_p = von_p[cell_sel]
            von_pref = von_pref[cell_sel]
            
            # Get matrices
            C0 = np.asarray(method_0['corr_matrix'])
            C1 = np.asarray(method_91['corr_matrix'])
            S = np.asarray(method_91['sparse'])
            
            # Compute partial correlations
            C1_inv = np.linalg.inv(C1)
            diag = np.sqrt(np.diag(C1_inv))
            partial_corr = -C1_inv / np.outer(diag, diag)
            np.fill_diagonal(partial_corr, 0)
            
            # Connectivity
            pos_connected = (S < 0) & (S != 0)
            neg_connected = S > 0
            
            # Tuned cells
            tuned = von_p < 0.05
            
            # Compute pairwise differences
            pairs = compute_pairwise_diffs(
                cell_xyz, von_pref, tuned, C0, partial_corr, pos_connected, neg_connected
            )
            
            all_data.append({
                'trace_set': ts,
                'label': site_info['label'],
                'pairs': pairs,
                'ncells': ncells,
            })
        except Exception:
            continue
    
    if not all_data:
        return html.Div([
            html.H1("Figure 7: Correlations and Connectivity vs Cell Pair Differences"),
            html.P("Data not yet available for the five high-connectivity sites.")
        ])
    
    # Define bins
    ori_bins = [0, 15, 45, 90]
    lateral_bins = [0, 25, 75, 150, 250]
    depth_bins = [0, 25, 60, 130]
    
    # Create 4x3 subplots
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            "Δori (tuned cells)", "Δx (pairs <30 µm in depth)", "Δz (pairs <30 µm laterally)",
            "", "", "",
            "", "", "",
            "", "", ""
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    axis_configs = [
        {
            'name': 'Δori (degrees)',
            'key': 'd_ori_deg',
            'bins': ori_bins,
            'mask_fn': lambda p: p['tuned_pairs'],
            'col': 1,
        },
        {
            'name': 'Δx (µm)',
            'key': 'd_lateral',
            'bins': lateral_bins,
            'mask_fn': lambda p: p['dz'] < 30,
            'col': 2,
        },
        {
            'name': 'Δz (µm)',
            'key': 'dz',
            'bins': depth_bins,
            'mask_fn': lambda p: p['d_lateral'] < 30,
            'col': 3,
        },
    ]
    
    connectivity_ratios = []
    
    # Plot each measure
    for config in axis_configs:
        col = config['col']
        
        for site_idx, site_data in enumerate(all_data):
            pairs = site_data['pairs']
            mask = config['mask_fn'](pairs)
            binned = bin_pairs(pairs, config['bins'], config['key'], mask)
            
            if len(binned['bin_centers']) == 0:
                continue
            
            x = binned['bin_centers']
            color = colors[site_idx]
            label = site_data['label']
            
            show_legend = (col == 3)  # Only show legend in rightmost column
            
            # Sample correlation (row 1)
            fig.add_trace(
                go.Scatter(
                    x=x, y=binned['sample_corr'],
                    mode='lines+markers',
                    line=dict(color=color),
                    marker=dict(size=6, color=color),
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend,
                    hovertemplate=f'{label}<br>%{{x}}<br>%{{y:.4f}}<extra></extra>'
                ),
                row=1, col=col
            )
            
            # Partial correlation (row 2)
            fig.add_trace(
                go.Scatter(
                    x=x, y=binned['partial_corr'],
                    mode='lines+markers',
                    line=dict(color=color),
                    marker=dict(size=6, color=color),
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=f'{label}<br>%{{x}}<br>%{{y:.4f}}<extra></extra>'
                ),
                row=2, col=col
            )
            
            # Positive connectivity (row 3)
            fig.add_trace(
                go.Scatter(
                    x=x, y=binned['frac_pos'],
                    mode='lines+markers',
                    line=dict(color='green'),
                    marker=dict(size=6, color='green', symbol='triangle-up'),
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=f'{label}<br>%{{x}}<br>%{{y:.3f}}<extra></extra>'
                ),
                row=3, col=col
            )
            
            # Negative connectivity (row 4)
            fig.add_trace(
                go.Scatter(
                    x=x, y=binned['frac_neg'],
                    mode='lines+markers',
                    line=dict(color='darkred'),
                    marker=dict(size=6, color='darkred', symbol='triangle-down'),
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=f'{label}<br>%{{x}}<br>%{{y:.3f}}<extra></extra>'
                ),
                row=4, col=col
            )
            
            # Store connectivity ratio for Δx analysis
            if config['key'] == 'd_lateral' and len(binned['bin_centers']) >= 4:
                bin_centers = np.array(binned['bin_centers'])
                frac_pos = np.array(binned['frac_pos'])
                
                close_idx = np.argmin(np.abs(bin_centers - 12.5))
                far_idx = np.argmin(np.abs(bin_centers - 200))
                
                if frac_pos[far_idx] > 0:
                    ratio = frac_pos[close_idx] / frac_pos[far_idx]
                    connectivity_ratios.append(ratio)
    
    # Update axes
    for col in range(1, 4):
        fig.update_xaxes(title_text=axis_configs[col-1]['name'], row=4, col=col)
    
    fig.update_yaxes(title_text="Sample Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Partial Correlation", row=2, col=1)
    fig.update_yaxes(title_text="Positive Connectivity", row=3, col=1)
    fig.update_yaxes(title_text="Negative Connectivity", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        **LAYOUT,
        height=1200,
        title_text=f"Correlations and Connectivity vs Cell Pair Differences ({len(all_data)} sites)",
        showlegend=True,
        legend=dict(x=1.02, y=0.98, xanchor='left', yanchor='top')
    )
    
    # Compute connectivity ratio statistic
    if connectivity_ratios:
        mean_ratio = float(np.mean(connectivity_ratios))
        ratio_text = html.Div([
            html.H3("Key Result"),
            html.P([
                f"Pairs under 25 µm apart are ",
                html.Strong(f"{mean_ratio:.2f}× more likely"),
                f" to be positively connected than pairs 150–250 µm apart ",
                f"(averaged over {len(connectivity_ratios)} sites)."
            ])
        ], className="stats")
    else:
        ratio_text = html.Div()
    
    return html.Div([
        html.H1("Figure 7: Correlations and Connectivity vs Cell Pair Differences"),
        html.P([
            f"Analysis of how sample correlations (method 0), partial correlations, and positive/negative "
            f"connectivity (method 91 sparse component) depend on orientation difference (Δori), lateral "
            f"distance (Δx), and depth difference (Δz) between cell pairs. Uses the five high-connectivity "
            f"sites (trace sets 30, 26, 28, 12, 45) with connectivity >3% in method 100. Only bins with "
            f"≥200 pairs are shown."
        ]),
        dcc.Graph(figure=fig),
        ratio_text
    ])
