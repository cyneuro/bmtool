"""
Plots for event-aligned analyses (``bmtool.analysis.events``): event shading, PETHs, event-aligned heatmaps,
paired before/after comparisons and per-node response differences.
"""

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats

__all__ = ["add_event_shading", "plot_peth", "plot_event_heatmap", "plot_paired_bars", "plot_node_differences"]


def add_event_shading(
    ax: plt.Axes,
    starts: Sequence[float],
    stops: Sequence[float],
    color: str = "0.9",
    alpha: float = 1.0,
    label: Optional[str] = None,
    zorder: float = 0,
) -> plt.Axes:
    """Shade ``[start, stop)`` spans (e.g. tones, shocks) behind the data; only the first span gets ``label``."""
    for k, (a, b) in enumerate(zip(starts, stops)):
        ax.axvspan(a, b, color=color, alpha=alpha, lw=0, zorder=zorder, label=label if k == 0 else None)
    return ax


def plot_peth(
    peth: xr.DataArray,
    ax: Optional[plt.Axes] = None,
    events: Optional[Sequence[Tuple[float, float]]] = None,
    event_color: str = "0.9",
    color: Optional[str] = None,
    label: Optional[str] = None,
    error: Optional[xr.DataArray] = None,
    xlabel: str = "time from event onset (ms)",
    ylabel: str = "rate (Hz)",
) -> plt.Axes:
    """
    Plot a PETH (``bmtool.analysis.events.peth``, dim ``lag``) as a line, optional shaded error band, and optional
    event spans ``[(start, stop), ...]`` relative to the onset.
    """
    if ax is None:
        _, ax = plt.subplots()
    lag = peth["lag"].values
    if events:
        add_event_shading(ax, [a for a, _ in events], [b for _, b in events], color=event_color)
    line = ax.plot(lag, peth.values, color=color, label=label)[0]
    if error is not None:
        ax.fill_between(lag, peth.values - error.values, peth.values + error.values, color=line.get_color(),
                        alpha=0.25, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_event_heatmap(
    x: xr.DataArray,
    ax: Optional[plt.Axes] = None,
    sort_window: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str = "z",
) -> plt.Axes:
    """
    Heatmap of a (node_id, lag) array (e.g. event-averaged z-scores), rows sorted by the mean over
    ``sort_window`` (descending) if given. Symmetric colour limits by default.
    """
    if ax is None:
        _, ax = plt.subplots()
    vals = np.asarray(x.transpose("node_id", "lag").values)
    lag = x["lag"].values
    if sort_window is not None:
        sel = (lag >= sort_window[0]) & (lag < sort_window[1])
        vals = vals[np.argsort(-np.nanmean(vals[:, sel], axis=1), kind="stable")]
    if vmin is None and vmax is None:
        lim = np.nanpercentile(np.abs(vals), 99) if vals.size else 1.0
        vmin, vmax = -lim, lim
    step = lag[1] - lag[0] if len(lag) > 1 else 1.0
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=(lag[0] - step / 2, lag[-1] + step / 2, len(vals), 0))
    plt.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_xlabel("time from event onset (ms)")
    ax.set_ylabel("node (sorted)" if sort_window is not None else "node")
    return ax


def plot_paired_bars(
    before: Sequence[float],
    after: Sequence[float],
    labels: Tuple[str, str] = ("before", "after"),
    stat: str = "mean",
    ax: Optional[plt.Axes] = None,
    colors: Tuple[str, str] = ("0.6", "C0"),
    n_resamples: int = 1000,
    seed: int = 0,
) -> Tuple[plt.Axes, Dict[str, float]]:
    """
    Bars of a paired per-node measure before vs after, with error bars (SEM for ``stat='mean'``, 95% bootstrap CI
    for ``stat='median'``) and a paired Wilcoxon signed-rank p-value above the bars.

    Returns the axes and {'before', 'after', 'p', 'frac_up'}.
    """
    from bmtool.analysis.ensembles import bootstrap_ci

    if ax is None:
        _, ax = plt.subplots()
    a, b = np.asarray(before, float), np.asarray(after, float)
    if stat == "mean":
        m = [a.mean(), b.mean()]
        err = np.array([[stats.sem(a), stats.sem(b)]] * 2)
    elif stat == "median":
        ca, cb = (bootstrap_ci(v, np.median, n_resamples=n_resamples, seed=seed) for v in (a, b))
        m = [ca[0], cb[0]]
        err = np.array([[ca[0] - ca[1], cb[0] - cb[1]], [ca[2] - ca[0], cb[2] - cb[0]]])
    else:
        raise ValueError("stat must be 'mean' or 'median'")
    p = float(stats.wilcoxon(b, a).pvalue)
    ax.bar([0, 1], m, color=list(colors), width=0.7)
    ax.errorbar([0, 1], m, yerr=err, fmt="none", ecolor="0.2", capsize=3, lw=1)
    top = max(m[0] + err[1, 0], m[1] + err[1, 1])
    ax.text(0.5, top * 1.05 if top > 0 else top + 0.05, "p<1e-4" if p < 1e-4 else f"p={p:.2g}", ha="center",
            va="bottom")
    ax.set_xticks([0, 1], list(labels))
    ax.set_ylabel(stat)
    return ax, {"before": float(m[0]), "after": float(m[1]), "p": p, "frac_up": float(np.mean(b > a))}


def plot_node_differences(
    diff: Sequence[float],
    groups: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    colors: Optional[Dict[str, str]] = None,
    ylabel: str = "difference",
) -> plt.Axes:
    """
    Per-node differences (e.g. z(after) - z(before)), sorted within each group and drawn as points with a zero line.
    """
    if ax is None:
        _, ax = plt.subplots()
    diff = np.asarray(diff, float)
    groups = np.asarray(groups) if groups is not None else np.array(["all"] * len(diff))
    x0 = 0
    for g in dict.fromkeys(groups):
        v = np.sort(diff[groups == g])
        ax.scatter(np.arange(x0, x0 + len(v)), v, s=6, color=(colors or {}).get(g), label=str(g))
        x0 += len(v) + max(5, len(diff) // 50)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("node (sorted within group)")
    ax.set_ylabel(ylabel)
    if len(set(groups)) > 1:
        ax.legend(frameon=False, markerscale=2)
    return ax
