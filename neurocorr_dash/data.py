"""
Read the neurocorr pipeline.

The dashboard reads the schema from the database (``dj.VirtualModule``), so it needs no
pipeline code. Schema names take the prefix in ``DATABASE_PREFIX``. Arrays are ``<npy@>``
values in the pipeline's object store (``DJ_STORES``), loaded with ``np.asarray``.
"""

from __future__ import annotations

import functools
import os

import datajoint as dj
import numpy as np
import pandas as pd

PREFIX = os.environ.get("DATABASE_PREFIX", "")
METHOD_NAMES = {
    0: "sample",
    1: "sample, evoked bins",
    2: "sample, first direction",
    3: "sample, second direction",
    10: "diag",
    30: "factor",
    80: "sparse",
    90: "sparse + latent",
    91: "sparse + latent, rank-capped",
    100: "sparse + latent",
}


@functools.cache
def covest():
    return dj.VirtualModule("covest", PREFIX + "covest")


def paper_site(aod_scan_start_time: int) -> int:
    """The site's number in the original code and figures: its scan start time mod 10000."""
    return int(aod_scan_start_time) % 10000


def site_name(session: int, site: int) -> str:
    """A site's short name in figures: its session and its number within it, as in "8-4"."""
    return f"{int(session)}-{int(site)}"


def site_label(row) -> str:
    """A site as the dashboard names it: its session and number, and its number in the paper."""
    return f"session {row.session}, site {row.site} ({row.paper_site} in the paper)"


def sites() -> pd.DataFrame:
    """One row per correlation set: its site, the cells recorded and kept, and the mean sample correlation."""
    c = covest()
    rows = (
        ((c.TrialTraceSet & {"high_repeats": True}) * c.Site)
        .proj("session", "site", "nneurons", "ndirs", "bin_ms", "aod_scan_start_time")
        .to_pandas()
        .reset_index()
    )
    active = c.CellSelection.proj("ncells").to_pandas().reset_index()
    rows = rows.merge(active, how="left")
    sample = c.CovMatrix & {"method": 0, "k": 1, "nfolds": 1}
    keys, matrices = sample.to_arrays("corr_matrix", include_key=True)
    mean_corr = {k["trace_set"]: mean_offdiagonal(np.asarray(R)) for k, R in zip(keys, matrices)}
    rows["mean_corr"] = rows["trace_set"].map(mean_corr)
    rows["paper_site"] = rows["aod_scan_start_time"].map(paper_site)
    rows["name"] = [site_name(a, b) for a, b in zip(rows["session"], rows["site"])]
    rows["analyzed"] = rows["ncells"] > 100
    return rows.sort_values(["session", "site"]).reset_index(drop=True)


def mean_offdiagonal(R: np.ndarray) -> float:
    return float(R[np.triu_indices_from(R, 1)].mean())


def validation_losses() -> pd.DataFrame:
    """Mean validation loss over the 10 folds, per site and method, for complete sites only."""
    c = covest()
    cv = ((c.CovMatrix & "nfolds = 10") * c.TrialTraceSet.proj("session", "site")).proj(
        "cv_loss", "session", "site"
    )
    cv = cv.to_pandas().reset_index()
    if cv.empty:
        return pd.DataFrame(columns=["site", "method", "cv_loss", "folds"])
    cv["site"] = [site_name(a, b) for a, b in zip(cv["session"], cv["site"])]
    grouped = cv.groupby(["site", "method"])["cv_loss"]
    losses = grouped.mean().to_frame("cv_loss").join(grouped.count().to_frame("folds")).reset_index()
    return losses[losses["folds"] == 10]


def progress() -> pd.DataFrame:
    """Per method: the estimates computed, of the 11 per analyzed site (10 folds and all trials)."""
    c = covest()
    done = c.CovMatrix.proj().to_pandas().reset_index().groupby("method").size()
    analyzed = (c.TrialTraceSet * c.CellSelection & "ncells > 100").proj("ndirs").to_pandas()
    methods = c.Method.proj("regularization", "condition_idx").to_pandas().reset_index()
    methods["name"] = methods["method"].map(METHOD_NAMES)
    methods["computed"] = methods["method"].map(done).fillna(0).astype(int)
    sites_per_method = np.where(
        methods["condition_idx"] == 0, len(analyzed), int((analyzed["ndirs"] == 2).sum())
    )
    methods["expected"] = 11 * sites_per_method
    return methods[["method", "name", "regularization", "computed", "expected"]]


def correlation_matrix(trace_set: int, method: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A correlation set's correlation matrix from all trials, and the positions of its active cells."""
    c = covest()
    key = {"trace_set": trace_set}
    R = (c.CovMatrix & key & {"method": method, "k": 1, "nfolds": 1}).fetch1("corr_matrix")
    xyz, selection = (c.TrialTraceSet * c.CellSelection & key).fetch1("cell_xyz", "selection")
    return np.asarray(R), np.asarray(xyz)[np.asarray(selection, dtype=bool)]
