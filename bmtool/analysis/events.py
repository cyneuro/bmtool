"""
Event-aligned spike analysis: counts in windows around events, binned rates, PETHs, baseline z-scores,
per-cell response classification and Repa et al. (2001) plasticity measures.

Conventions
-----------
- Times are in milliseconds (as in BMTK spike files and ``load_spikes_to_df``).
- Every window is half-open, ``[start, stop)``.
- ``spike_df`` is a DataFrame with ``node_ids`` and ``timestamps`` columns (``load_spikes_to_df``).
- Cell-by-event and cell-by-time results are ``xarray.DataArray`` objects with a ``node_id`` dimension.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

__all__ = [
    "spike_arrays",
    "count_spikes_in_windows",
    "bin_spikes",
    "event_aligned",
    "peth",
    "baseline_zscore",
    "classify_response",
    "population_response_z",
    "population_percent_change",
    "population_rate_change",
    "response_auc",
    "repa_zscores",
    "repa_max_block_difference",
    "repa_plastic_cells",
]


# ----------------------------------------------------------------------------- basics
def spike_arrays(
    spike_df: pd.DataFrame, node_col: str = "node_ids", time_col: str = "timestamps"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (node_ids, timestamps) as numpy arrays sorted by time (stable)."""
    ids = np.asarray(spike_df[node_col]).astype(np.int64)
    ts = np.asarray(spike_df[time_col], dtype=float)
    order = np.argsort(ts, kind="stable")
    return ids[order], ts[order]


def _row_index(ids: np.ndarray, node_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Row of each spike in ``node_ids`` (sorted lookup) and a mask of spikes whose node is in ``node_ids``."""
    order = np.argsort(node_ids, kind="stable")
    sorted_ids = node_ids[order]
    pos = np.searchsorted(sorted_ids, ids)
    pos_c = np.clip(pos, 0, len(sorted_ids) - 1)
    ok = (pos < len(sorted_ids)) & (sorted_ids[pos_c] == ids) if len(sorted_ids) else np.zeros(len(ids), bool)
    return order[pos_c], ok


def _node_ids(ids: np.ndarray, node_ids: Optional[Sequence[int]]) -> np.ndarray:
    return np.unique(ids) if node_ids is None else np.asarray(node_ids, dtype=np.int64)


def count_spikes_in_windows(
    spike_df: pd.DataFrame,
    onsets: Sequence[float],
    window: Tuple[float, float],
    node_ids: Optional[Sequence[int]] = None,
) -> xr.DataArray:
    """
    Spike counts of each node in ``[onset + window[0], onset + window[1])`` for every onset.

    Parameters
    ----------
    spike_df : pd.DataFrame
        Spikes with ``node_ids`` and ``timestamps`` (ms).
    onsets : sequence of float
        Event onsets (ms).
    window : (float, float)
        Window start and stop relative to each onset (ms).
    node_ids : sequence of int, optional
        Rows of the result, in this order. Default: nodes that spiked, sorted. Nodes without spikes get zeros.

    Returns
    -------
    xr.DataArray
        Counts with dims (node_id, event); coords ``onset`` on the event dimension.
    """
    ids, ts = spike_arrays(spike_df)
    nodes = _node_ids(ids, node_ids)
    row, ok = _row_index(ids, nodes)
    onsets = np.asarray(onsets, dtype=float)
    out = np.zeros((len(nodes), len(onsets)))
    for j, t0 in enumerate(onsets):
        lo, hi = np.searchsorted(ts, [t0 + window[0], t0 + window[1]])
        sel = ok[lo:hi]
        np.add.at(out[:, j], row[lo:hi][sel], 1)
    return xr.DataArray(
        out,
        dims=("node_id", "event"),
        coords={"node_id": nodes, "event": np.arange(len(onsets)), "onset": ("event", onsets)},
        attrs={"window": tuple(window), "units": "spikes"},
    )


def bin_spikes(
    spike_df: pd.DataFrame,
    bin_size: float,
    t_start: float = 0.0,
    t_stop: Optional[float] = None,
    node_ids: Optional[Sequence[int]] = None,
    rate: bool = False,
    dtype=np.float64,
) -> xr.DataArray:
    """
    Spike counts (or rates in Hz) per node in consecutive bins from ``t_start``.

    Bin edges are ``t_start + k * bin_size`` up to the first edge >= ``t_stop``; spikes outside the edges are
    dropped (numpy histogram semantics: half-open bins, last bin closed).

    Returns
    -------
    xr.DataArray
        dims (node_id, time); ``time`` = bin centres (ms); ``attrs['edges']`` = bin edges.
    """
    ids, ts = spike_arrays(spike_df)
    if t_stop is None:
        t_stop = ts.max() if len(ts) else t_start + bin_size
    edges = np.arange(t_start, t_stop + bin_size, bin_size)
    nodes = _node_ids(ids, node_ids)
    row, ok = _row_index(ids, nodes)
    counts, _, _ = np.histogram2d(row[ok], ts[ok], bins=(np.arange(len(nodes) + 1) - 0.5, edges))
    counts = counts.astype(dtype)
    if rate:
        counts /= bin_size / 1000.0
    return xr.DataArray(
        counts,
        dims=("node_id", "time"),
        coords={"node_id": nodes, "time": edges[:-1] + bin_size / 2},
        attrs={"edges": edges, "bin_size": bin_size, "units": "Hz" if rate else "spikes"},
    )


def event_aligned(binned: xr.DataArray, onsets: Sequence[float], window: Tuple[float, float]) -> xr.DataArray:
    """
    Cut a binned (node_id, time) array into event-aligned segments.

    The segment of each onset starts at the bin whose left edge is closest to ``onset + window[0]`` and has
    ``round((window[1] - window[0]) / bin_size)`` bins.

    Returns
    -------
    xr.DataArray
        dims (node_id, event, lag); ``lag`` = bin left edge relative to the onset (ms).
    """
    bs = float(binned.attrs["bin_size"])
    edges = np.asarray(binned.attrs["edges"])
    n = int(round((window[1] - window[0]) / bs))
    segs = []
    for t0 in onsets:
        k = int(round((t0 + window[0] - edges[0]) / bs))
        if k < 0 or k + n > binned.sizes["time"]:
            raise ValueError(f"event at {t0} ms: window {window} runs past the binned range")
        segs.append(binned.values[:, k:k + n])
    lag = window[0] + bs * np.arange(n)
    return xr.DataArray(
        np.stack(segs, axis=1),
        dims=("node_id", "event", "lag"),
        coords={"node_id": binned["node_id"].values, "event": np.arange(len(onsets)),
                "onset": ("event", np.asarray(onsets, float)), "lag": lag},
        attrs=dict(binned.attrs),
    )


def peth(
    spike_df: pd.DataFrame,
    onsets: Sequence[float],
    window: Tuple[float, float],
    bin_size: float,
    node_ids: Optional[Sequence[int]] = None,
    per_node: bool = False,
) -> xr.DataArray:
    """
    Peri-event time histogram in Hz, averaged over events (and over nodes unless ``per_node``).

    Bins are half-open ``[lag, lag + bin_size)`` with ``lag`` from ``window[0]`` to the last edge <= ``window[1]``.
    The population PETH divides by ``len(node_ids)`` (nodes without spikes count as zeros).

    Returns
    -------
    xr.DataArray
        dims (lag,) or (node_id, lag); ``lag`` = bin centres relative to the onset (ms).
    """
    lefts = np.arange(window[0], window[1] - bin_size + 1e-9 * abs(bin_size), bin_size)
    counts = np.stack(
        [count_spikes_in_windows(spike_df, onsets, (lo, lo + bin_size), node_ids).sum("event").values
         for lo in lefts],
        axis=1,
    )
    nodes = _node_ids(np.asarray(spike_df["node_ids"]).astype(np.int64), node_ids)
    hz = counts / (len(onsets) * bin_size / 1000.0)
    lag = lefts + bin_size / 2
    if per_node:
        return xr.DataArray(hz, dims=("node_id", "lag"), coords={"node_id": nodes, "lag": lag}, attrs={"units": "Hz"})
    return xr.DataArray(hz.sum(0) / len(nodes), dims=("lag",), coords={"lag": lag}, attrs={"units": "Hz per node"})


def baseline_zscore(
    x: xr.DataArray,
    baseline,
    dim: str = "time",
    sd_floor: Optional[float] = None,
) -> xr.DataArray:
    """
    z-score each node against its own baseline samples along ``dim``.

    Parameters
    ----------
    x : xr.DataArray
        Data with a ``node_id`` dimension and ``dim``.
    baseline : array-like of bool or int
        Boolean mask or integer positions along ``dim`` that form the baseline.
    sd_floor : float, optional
        Lower bound for the baseline SD (e.g. the Poisson SD of a 1 Hz cell for one bin). Without a floor, nodes
        with zero baseline SD get NaN.
    """
    base = x.isel({dim: np.asarray(baseline)})
    mu = base.mean(dim)
    sd = base.std(dim)
    sd = np.maximum(sd, sd_floor) if sd_floor is not None else sd.where(sd != 0)
    z = (x - mu) / sd
    z.attrs = {**x.attrs, "units": "z"}
    return z


# ----------------------------------------------------------------------------- per-cell classification
def classify_response(
    resp,
    base,
    alpha: float = 0.05,
    test: str = "poisson",
    resp_duration: Optional[float] = None,
    base_duration: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-node Up / Down / None classification of a response against a baseline.

    Parameters
    ----------
    resp, base : array-like
        1-D total counts per node (``test='poisson'``), or 2-D (node, sample) counts per bin or trial.
    test : {'poisson', 'ranksum'}
        ``poisson``: exact conditional test of two Poisson rates on the summed counts,
        ``n_resp ~ Binomial(n_resp + n_base, T_resp / (T_resp + T_base))`` under H0 (``scipy.stats.binomtest``);
        keeps its power for sparse counts. ``ranksum``: two-sided Mann-Whitney U on the per-sample counts (2-D input).
    resp_duration, base_duration : float, optional
        Durations of the summed windows (any unit). Default: the number of samples (2-D input).

    Returns
    -------
    classes : np.ndarray of str ('Up', 'Down', 'None')
    pvalues : np.ndarray of float (NaN where a node has no spikes)
    """
    resp, base = np.asarray(resp, dtype=float), np.asarray(base, dtype=float)
    two_d = resp.ndim == 2
    if resp_duration is None or base_duration is None:
        if not two_d:
            raise ValueError("durations are required for 1-D (summed) counts")
        resp_duration, base_duration = resp.shape[1], base.shape[1]
    r_tot = resp.sum(1) if two_d else resp
    b_tot = base.sum(1) if two_d else base
    frac = resp_duration / (resp_duration + base_duration)
    classes = np.full(len(r_tot), "None", dtype=object)
    pvals = np.full(len(r_tot), np.nan)
    for i, (r, b) in enumerate(zip(r_tot, b_tot)):
        if r + b == 0:
            continue
        if test == "poisson":
            p = stats.binomtest(int(r), int(r + b), frac).pvalue
            up = r / resp_duration > b / base_duration
        elif test == "ranksum":
            if not two_d:
                raise ValueError("test='ranksum' needs per-sample (2-D) counts")
            p = stats.mannwhitneyu(resp[i], base[i], alternative="two-sided").pvalue
            up = resp[i].mean() > base[i].mean()
        else:
            raise ValueError(f"unknown test {test!r}")
        pvals[i] = p
        if p < alpha:
            classes[i] = "Up" if up else "Down"
    return classes.astype(str), pvals


# ----------------------------------------------------------------------------- population responses
def population_response_z(
    spike_df: pd.DataFrame,
    onsets: Sequence[float],
    window: Tuple[float, float],
    baseline_window: Tuple[float, float],
    bin_size: float,
    node_ids: Sequence[int],
    baseline_onsets: Optional[Sequence[float]] = None,
) -> float:
    """
    Population z of the summed response of ``node_ids`` in ``window`` (expressed per ``bin_size`` bin) against the
    distribution of summed population counts in the ``bin_size`` bins of ``baseline_window`` before each
    ``baseline_onsets`` (default ``onsets``). Returns NaN for empty inputs or zero baseline SD.
    """
    if len(node_ids) == 0 or len(onsets) == 0:
        return np.nan
    bon = onsets if baseline_onsets is None else baseline_onsets
    nb = int(round((baseline_window[1] - baseline_window[0]) / bin_size))
    base_bins = np.concatenate([
        count_spikes_in_windows(spike_df, bon, (baseline_window[0] + k * bin_size,
                                                baseline_window[0] + (k + 1) * bin_size), node_ids).values.sum(0)
        for k in range(nb)])
    resp = count_spikes_in_windows(spike_df, onsets, window, node_ids).values.sum(0) * (
        bin_size / (window[1] - window[0]))
    sd = base_bins.std()
    return float((resp.mean() - base_bins.mean()) / sd) if sd > 0 else np.nan


def _pop_rates(spike_df, onsets, window, baseline_window, node_ids, baseline_onsets):
    bon = onsets if baseline_onsets is None else baseline_onsets
    r = count_spikes_in_windows(spike_df, onsets, window, node_ids).values
    r0 = count_spikes_in_windows(spike_df, bon, baseline_window, node_ids).values
    return r, r0, bon


def population_percent_change(
    spike_df: pd.DataFrame,
    onsets: Sequence[float],
    window: Tuple[float, float],
    baseline_window: Tuple[float, float],
    node_ids: Sequence[int],
    baseline_onsets: Optional[Sequence[float]] = None,
) -> float:
    """Percent change of the population rate in ``window`` vs ``baseline_window`` (per-event mean rates)."""
    if len(node_ids) == 0 or len(onsets) == 0:
        return np.nan
    r, r0, bon = _pop_rates(spike_df, onsets, window, baseline_window, node_ids, baseline_onsets)
    rate = r.sum() / (window[1] - window[0]) / len(onsets)
    rate0 = r0.sum() / (baseline_window[1] - baseline_window[0]) / len(bon)
    return float(100.0 * (rate / rate0 - 1.0)) if rate0 > 0 else np.nan


def population_rate_change(
    spike_df: pd.DataFrame,
    onsets: Sequence[float],
    window: Tuple[float, float],
    baseline_window: Tuple[float, float],
    node_ids: Sequence[int],
    baseline_onsets: Optional[Sequence[float]] = None,
) -> float:
    """Mean per-node rate increase (Hz) in ``window`` over ``baseline_window`` (windows in ms)."""
    r, r0, _ = _pop_rates(spike_df, onsets, window, baseline_window, node_ids, baseline_onsets)
    return float(r.mean() / ((window[1] - window[0]) / 1000.0)
                 - r0.mean() / ((baseline_window[1] - baseline_window[0]) / 1000.0))


def response_auc(x: xr.DataArray, window: Tuple[float, float], dim: str = "lag") -> xr.DataArray:
    """Mean of ``x`` over ``window`` (half-open, on the ``dim`` coordinate) times the window length (ms)."""
    c = x[dim].values
    sel = (c >= window[0]) & (c < window[1])
    return x.isel({dim: np.where(sel)[0]}).mean(dim) * (window[1] - window[0])


# ----------------------------------------------------------------------------- Repa et al. (2001)
def repa_zscores(
    spike_df: pd.DataFrame,
    onsets: Sequence[float],
    baseline_events: Sequence[int],
    response_window: Tuple[float, float] = (0.0, 200.0),
    bin_size: float = 10.0,
    baseline_duration: float = 200.0,
    node_ids: Optional[Sequence[int]] = None,
) -> xr.DataArray:
    """
    Per-trial average z-scores (Repa et al. 2001).

    Baseline mean and SD (population SD, ddof=0) of each node's counts in ``bin_size`` bins of the
    ``baseline_duration`` before every onset in ``baseline_events`` (indices into ``onsets``); per trial, the
    response-window bins are z-scored and averaged. Nodes with zero baseline SD get z = 0.

    Returns
    -------
    xr.DataArray
        dims (node_id, event); attrs ``mu`` / ``sigma`` as coordinates on node_id.
    """
    onsets = np.asarray(onsets, float)
    base_on = onsets[np.asarray(baseline_events, dtype=int)]
    base = np.concatenate([
        count_spikes_in_windows(spike_df, base_on, (lo, lo + bin_size), node_ids).values
        for lo in np.arange(-baseline_duration, 0.0, bin_size)], axis=1)
    mu, sigma = base.mean(1), base.std(1)
    lefts = np.arange(response_window[0], response_window[1], bin_size)
    resp = np.stack([count_spikes_in_windows(spike_df, onsets, (lo, lo + bin_size), node_ids).values
                     for lo in lefts], axis=2)  # node, event, bin
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sigma[:, None, None] > 0, (resp - mu[:, None, None]) / sigma[:, None, None], 0.0).mean(2)
    nodes = count_spikes_in_windows(spike_df, onsets[:1], (0, 1), node_ids)["node_id"].values
    return xr.DataArray(z, dims=("node_id", "event"),
                        coords={"node_id": nodes, "event": np.arange(len(onsets)), "onset": ("event", onsets),
                                "mu": ("node_id", mu), "sigma": ("node_id", sigma)})


def _blocks(values, block_size, partial=True):
    n = len(values) // block_size
    out = [values[k * block_size:(k + 1) * block_size] for k in range(n)]
    if partial and len(values) > n * block_size:
        out.append(values[n * block_size:])
    return out


def repa_max_block_difference(
    z: xr.DataArray, baseline_events: Sequence[int], conditioning_events: Sequence[int], block_size: int = 4
) -> pd.DataFrame:
    """
    Repa Fig. 4: per node, mean z of each conditioning block (``block_size`` trials; a remainder forms a last
    partial block) minus the mean baseline z; the maximum over blocks is ``max_z_difference``.
    """
    zb = z.values[:, np.asarray(baseline_events, int)]
    zc = z.values[:, np.asarray(conditioning_events, int)]
    rows = []
    for i, node in enumerate(z["node_id"].values):
        zb_mean = np.mean(zb[i])
        diffs = [np.mean(b) - zb_mean for b in _blocks(list(zc[i]), block_size)]
        rows.append({"node_id": node, "z_avg_baseline": zb_mean,
                     "max_z_difference": max(diffs) if diffs else np.nan,
                     "z_diff_per_block": diffs, "n_blocks": len(diffs)})
    return pd.DataFrame(rows)


def repa_plastic_cells(
    z: xr.DataArray,
    baseline_events: Sequence[int],
    conditioning_events: Sequence[int],
    block_size: int = 4,
    alpha: float = 0.05,
    min_sig_blocks: int = 1,
    persistence_threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Repa et al. (2001) plastic cells: a node is plastic if at least ``min_sig_blocks`` full conditioning blocks
    exceed the baseline z (one-tailed Welch-free t-test, ``scipy.stats.ttest_ind(..., alternative='greater')``).
    For plastic nodes with >= 16 conditioning trials, persistence = (late 8 - baseline) / (early 8 - baseline);
    ``long_lasting`` if >= ``persistence_threshold``, else ``transient``.
    """
    zb = z.values[:, np.asarray(baseline_events, int)]
    zc = z.values[:, np.asarray(conditioning_events, int)]
    rows = []
    for i, node in enumerate(z["node_id"].values):
        n_sig = 0
        for blk in _blocks(list(zc[i]), block_size, partial=False):
            if len(blk) < 2:
                continue
            p = stats.ttest_ind(blk, zb[i], alternative="greater").pvalue
            if np.isfinite(p) and p < alpha:
                n_sig += 1
        plastic = n_sig >= min_sig_blocks
        persistence, cell_type = np.nan, np.nan
        if plastic and zc.shape[1] >= 16:
            zb_mean = np.mean(zb[i])
            early, late = np.mean(zc[i, :8]) - zb_mean, np.mean(zc[i, 8:16]) - zb_mean
            if early != 0:
                persistence = late / early
            cell_type = "long_lasting" if persistence >= persistence_threshold else "transient"
        rows.append({"node_id": node, "is_plastic": plastic, "n_sig_blocks": n_sig,
                     "persistence": persistence, "cell_type": cell_type})
    return pd.DataFrame(rows)
