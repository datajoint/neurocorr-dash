"""Figures from synthetic inputs; no database needed."""

import numpy as np
import pandas as pd

from neurocorr_dash import data, figures


def test_site_label():
    assert data.site_label(3453748112031) == 2031


def test_heatmap_blanks_the_diagonal():
    R = np.eye(5) + 0.1 * (1 - np.eye(5))
    fig = figures.correlation_heatmap(R, "test")
    z = np.array(fig.data[0].z, dtype=float)
    assert np.isnan(np.diag(z)).all() and np.allclose(z[0, 1], 0.1)


def test_sites_scatter_shows_analyzed_sites_only():
    sites = pd.DataFrame(
        {"ncells": [150, 90], "mean_corr": [0.02, 0.01], "site": [1, 2], "analyzed": [True, False]}
    )
    fig = figures.sites_scatter(sites)
    assert list(fig.data[0].x) == [150]


def test_loss_differences_pairs_sites():
    losses = pd.DataFrame(
        {"site": [1, 1, 2, 2, 3], "method": [0, 10, 0, 10, 0], "cv_loss": [0.62, 0.60, 0.70, 0.66, 0.5]}
    )
    fig = figures.loss_differences(losses, reference=10)
    assert len(fig.data) == 1
    np.testing.assert_allclose(sorted(fig.data[0].y), [0.02, 0.04])


def test_loss_differences_without_the_reference_is_empty():
    losses = pd.DataFrame({"site": [1], "method": [0], "cv_loss": [0.6]})
    assert len(figures.loss_differences(losses, reference=10).data) == 0


def test_layouts_accept_extra_query_parameters():
    """A host page may add query parameters (a token, a theme); Dash passes them to layout()."""
    import inspect

    import dash

    import neurocorr_dash.app  # noqa: F401  registers the pages

    layouts = [page["layout"] for page in dash.page_registry.values()]
    assert len(layouts) == 4
    for layout in layouts:
        assert any(p.kind is p.VAR_KEYWORD for p in inspect.signature(layout).parameters.values())
