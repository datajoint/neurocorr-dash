"""
Read the neurocorr pipeline.

The dashboard reads the schema from the database (``dj.VirtualModule``), so it needs no
pipeline code. Schema names take the prefix in ``DATABASE_PREFIX``.
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


def site_label(aod_scan_start_time: int) -> int:
    """The site number used in the original code and figures: the scan's start time mod 10000."""
    return int(aod_scan_start_time) % 10000


def sites() -> pd.DataFrame:
    """One row per correlation set: the cells recorded and kept, and the mean sample correlation."""
    c = covest()
    rows = (c.Traces & {"high_repeats": True}).proj("nneurons", "ndirs", "bin_ms").to_pandas().reset_index()
    active = c.ActiveCells.proj("ncells").to_pandas().reset_index()
    rows = rows.merge(active, how="left")
    sample = c.CovMatrix & {"method": 0, "k": 1, "nfolds": 1}
    keys, matrices = sample.to_arrays("corr_matrix", include_key=True)
    mean_corr = {k["aod_scan_start_time"]: mean_offdiagonal(R) for k, R in zip(keys, matrices)}
    rows["mean_corr"] = rows["aod_scan_start_time"].map(mean_corr)
    rows["site"] = rows["aod_scan_start_time"].map(site_label)
    rows["analyzed"] = rows["ncells"] > 100
    return rows.sort_values("aod_scan_start_time").reset_index(drop=True)


def mean_offdiagonal(R: np.ndarray) -> float:
    return float(R[np.triu_indices_from(R, 1)].mean())


def validation_losses() -> pd.DataFrame:
    """Mean validation loss over the 10 folds, per site and method, for complete sites only."""
    c = covest()
    cv = (c.CovMatrix & "nfolds = 10").proj("cv_loss").to_pandas().reset_index()
    if cv.empty:
        return pd.DataFrame(columns=["aod_scan_start_time", "method", "cv_loss", "folds", "site"])
    grouped = cv.groupby(["aod_scan_start_time", "method"])["cv_loss"]
    losses = grouped.mean().to_frame("cv_loss").join(grouped.count().to_frame("folds")).reset_index()
    losses = losses[losses["folds"] == 10]
    losses["site"] = losses["aod_scan_start_time"].map(site_label)
    return losses


def progress() -> pd.DataFrame:
    """Per method: the estimates computed, of the 11 per analyzed site (10 folds and all trials)."""
    c = covest()
    done = c.CovMatrix.proj().to_pandas().reset_index().groupby("method").size()
    analyzed = (c.Traces * c.ActiveCells & "ncells > 100").proj("ndirs").to_pandas()
    methods = c.Method.proj("regularization", "condition_idx").to_pandas().reset_index()
    methods["name"] = methods["method"].map(METHOD_NAMES)
    methods["computed"] = methods["method"].map(done).fillna(0).astype(int)
    sites_per_method = np.where(
        methods["condition_idx"] == 0, len(analyzed), int((analyzed["ndirs"] == 2).sum())
    )
    methods["expected"] = 11 * sites_per_method
    return methods[["method", "name", "regularization", "computed", "expected"]]


def correlation_matrix(aod_scan_start_time: int, method: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A site's correlation matrix from all trials, and the positions of its active cells."""
    c = covest()
    key = {"aod_scan_start_time": aod_scan_start_time}
    R = (c.CovMatrix & key & {"method": method, "k": 1, "nfolds": 1}).fetch1("corr_matrix")
    xyz, selection = (c.Traces * c.ActiveCells & key).fetch1("cell_xyz", "selection")
    return R, np.asarray(xyz)[np.asarray(selection, dtype=bool)]
