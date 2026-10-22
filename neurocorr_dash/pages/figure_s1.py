import dash
from dash import html, dcc
import numpy as np
import plotly.graph_objects as go

from neurocorr_dash import data
from neurocorr_dash.figures import LAYOUT

dash.register_page(__name__, path="/figure-s1", name="Figure S1", order=10)


def layout(**_query):
    try:
        c = data.covest()
    except Exception as error:
        return html.P(f"The pipeline is not available: {error}")
    
    # Fetch the specific CovMatrix row: trace_set=28, method=100, k=1, nfolds=1
    query = c.CovMatrix & {'trace_set': 28, 'method': 100, 'k': 1, 'nfolds': 1}
    
    if not query:
        return html.Div([
            html.H1("Figure S1: Hyperparameter Search Landscape"),
            html.P("Waiting for trace_set 28, method 100 to complete.")
        ])
    
    # Fetch the arrays
    visited = np.asarray(query.fetch1('visited'))
    losses = np.asarray(query.fetch1('losses'))
    hypers = np.asarray(query.fetch1('hypers'))
    
    # Get the Method row for hyperparam_space
    method_row = (c.Method & {'method': 100}).fetch1()
    hyperparam_space = method_row['hyperparam_space']
    
    # Extract the grid definitions
    alpha_spec = hyperparam_space[0]  # {"log": [-7, -1.5, 0.05], "name": "alpha"}
    beta_spec = hyperparam_space[1]   # {"log": [-5, -1.5, 0.05], "name": "beta"}
    
    # Generate the alpha and beta grids
    alpha_log_min, alpha_log_max, alpha_log_step = alpha_spec['log']
    beta_log_min, beta_log_max, beta_log_step = beta_spec['log']
    
    # Number of grid points in each dimension
    n_alpha = int((alpha_log_max - alpha_log_min) / alpha_log_step) + 1
    n_beta = int((beta_log_max - beta_log_min) / beta_log_step) + 1
    
    # Generate grids: α = exp(-7 + 0.05·i), β = exp(-5 + 0.05·j)
    alpha_grid = np.exp(alpha_log_min + alpha_log_step * np.arange(n_alpha))
    beta_grid = np.exp(beta_log_min + beta_log_step * np.arange(n_beta))
    
    chosen_alpha = float(hypers[0])
    chosen_beta = float(hypers[1])
    n_visited = len(visited)
    
    # Compute the loss minus its minimum (in nats)
    min_loss = float(losses.min())
    relative_losses = losses - min_loss
    
    # Get the visited index range
    i_min, i_max = int(visited[:, 0].min()), int(visited[:, 0].max())
    j_min, j_max = int(visited[:, 1].min()), int(visited[:, 1].max())
    
    # Create a 2D grid for the losses, filled with NaN
    loss_grid = np.full((j_max - j_min + 1, i_max - i_min + 1), np.nan)
    
    # Fill in the visited points
    for (i, j), loss in zip(visited, relative_losses):
        loss_grid[int(j) - j_min, int(i) - i_min] = loss
    
    # Create log-scale axes for the grid
    log_alpha_vals = np.log(alpha_grid[i_min:i_max + 1])
    log_beta_vals = np.log(beta_grid[j_min:j_max + 1])
    
    # Create the contour plot
    fig = go.Figure()
    
    # Add contour
    fig.add_trace(go.Contour(
        z=loss_grid,
        x=log_alpha_vals,
        y=log_beta_vals,
        colorscale='Viridis',
        connectgaps=True,
        colorbar={'title': 'Loss - min(Loss)<br>(nats)'},
        contours={'coloring': 'heatmap'},
        hovertemplate='log α: %{x:.2f}<br>log β: %{y:.2f}<br>Δ loss: %{z:.3f} nats<extra></extra>',
        name='Loss landscape'
    ))
    
    # Add scatter points for visited locations
    visited_log_alpha = np.log(alpha_grid[visited[:, 0]])
    visited_log_beta = np.log(beta_grid[visited[:, 1]])
    
    fig.add_trace(go.Scatter(
        x=visited_log_alpha,
        y=visited_log_beta,
        mode='markers',
        marker={'size': 4, 'color': 'white', 'symbol': 'circle', 'line': {'width': 0.5, 'color': 'black'}},
        hovertemplate='Visited point<br>log α: %{x:.2f}<br>log β: %{y:.2f}<extra></extra>',
        name='Visited points',
        showlegend=True
    ))
    
    # Add chosen hyperparameters as red cross
    chosen_log_alpha = np.log(chosen_alpha)
    chosen_log_beta = np.log(chosen_beta)
    
    fig.add_trace(go.Scatter(
        x=[chosen_log_alpha],
        y=[chosen_log_beta],
        mode='markers',
        marker={'size': 15, 'color': 'red', 'symbol': 'cross', 'line': {'width': 2}},
        hovertemplate=f'Chosen<br>α = {chosen_alpha:.4f}<br>β = {chosen_beta:.4f}<extra></extra>',
        name=f'Chosen: α={chosen_alpha:.4f}, β={chosen_beta:.4f}',
        showlegend=True
    ))
    
    fig.update_layout(
        **LAYOUT,
        title=f'Panel A: Cross-validated loss landscape (session 8828, site 2031)',
        xaxis_title='log α',
        yaxis_title='log β',
        height=600,
        legend={'x': 0.5, 'y': -0.15, 'xanchor': 'center', 'yanchor': 'top', 'orientation': 'h'}
    )
    
    return html.Div([
        html.H1("Figure S1: Hyperparameter Search"),
        html.P(
            f"Loss landscape for the sparse+latent estimator at session 8828, site 2031 "
            f"(trace set 28, method 100). The hyperparameter search visited {n_visited} points "
            f"on a grid defined by α = exp(−7 + 0.05·i) and β = exp(−5 + 0.05·j). "
            f"The plot shows the mean inner validation loss minus its minimum, in nats. "
            f"The chosen hyperparameters are α = {chosen_alpha:.4f} and β = {chosen_beta:.4f}."
        ),
        dcc.Graph(figure=fig)
    ])
