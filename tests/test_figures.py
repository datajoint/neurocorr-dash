"""Figures from synthetic inputs; no database needed."""

import numpy as np
import pandas as pd

from neurocorr_dash import data, figures


def test_site_labels():
    assert data.paper_site(3453748112031) == 2031
    assert data.site_name(8, 4) == "8-4"
    row = pd.Series({"session": 8, "site": 4, "paper_site": 2031})
    assert data.site_label(row) == "session 8, site 4 (2031 in the paper)"


def test_heatmap_blanks_the_diagonal():
    R = np.eye(5) + 0.1 * (1 - np.eye(5))
    fig = figures.correlation_heatmap(R, "test")
    z = np.array(fig.data[0].z, dtype=float)
    assert np.isnan(np.diag(z)).all() and np.allclose(z[0, 1], 0.1)


def test_sites_scatter_shows_analyzed_sites_only():
    sites = pd.DataFrame(
        {
            "ncells": [150, 90],
            "mean_corr": [0.02, 0.01],
            "name": ["8-4", "8-5"],
            "analyzed": [True, False],
        }
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
    assert len(layouts) == 5
    for layout in layouts:
        assert any(p.kind is p.VAR_KEYWORD for p in inspect.signature(layout).parameters.values())


def test_fig3_panels():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 30)) + 0.3 * rng.standard_normal((500, 1))
    R = np.corrcoef(X.T)
    e = figures.fig3e(R, "session 8, site 4")
    assert e.data[0].zmin == -0.25 and np.isnan(np.diag(np.array(e.data[0].z, dtype=float))).all()
    f = figures.fig3f(R, "session 8, site 4")
    assert len(f.data[0].x) == 30 * 29 // 2
    assert "session 8, site 4" in e.layout.title.text
