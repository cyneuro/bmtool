"""
Population (ensemble) structure of neural responses: pairwise similarity of response vectors, per-node similarity
summaries with bootstrap confidence intervals, and PCA trajectories across conditions.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = ["pairwise_cosine_similarity", "per_node_similarity", "bootstrap_ci", "trajectory_pca"]


def pairwise_cosine_similarity(x: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every pair of rows of ``x`` (nodes x features), s_ij = x_i.x_j / (|x_i| |x_j|).

    Uses ``sklearn.metrics.pairwise.cosine_similarity`` when scikit-learn is installed, otherwise numpy. Rows with
    zero norm should be removed first (their similarity is undefined).
    """
    x = np.asarray(x)
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:  # optional dependency
        xn = x / np.linalg.norm(x, axis=1, keepdims=True)
        return xn @ xn.T
    return cosine_similarity(x)


def per_node_similarity(s: np.ndarray, stat: str = "mean") -> np.ndarray:
    """
    Each node's mean or median similarity to all *other* nodes (diagonal excluded) from a square matrix ``s``.
    """
    s = np.asarray(s)
    n = s.shape[0]
    off = s[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    if stat == "mean":
        return off.mean(1)
    if stat == "median":
        return np.median(off, axis=1)
    raise ValueError(f"stat must be 'mean' or 'median', not {stat!r}")


def bootstrap_ci(
    x: Sequence[float],
    statistic: Callable = np.median,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    method: str = "percentile",
    seed: Optional[int] = 0,
) -> Tuple[float, float, float]:
    """
    Statistic of ``x`` and its bootstrap confidence interval (``scipy.stats.bootstrap``, resampling the elements of
    ``x``). Returns (statistic, low, high).
    """
    x = np.asarray(x, dtype=float)
    res = stats.bootstrap((x,), lambda a, axis: statistic(a, axis=axis), n_resamples=n_resamples,
                          confidence_level=confidence, method=method, random_state=seed, vectorized=True)
    return float(statistic(x)), float(res.confidence_interval.low), float(res.confidence_interval.high)


def trajectory_pca(
    traces: Dict[str, np.ndarray],
    n_components: int = 3,
    fit_on: Optional[Sequence[str]] = None,
    sign_reference: Optional[Tuple[str, slice]] = None,
):
    """
    PCA of trial-averaged population activity pooled across conditions, with each condition projected on the
    common components (population trajectories).

    Parameters
    ----------
    traces : dict
        {condition: array (nodes x time)}; all with the same nodes.
    fit_on : list of str, optional
        Conditions concatenated along time for the fit (default: all, in dict order). Only these are projected.
    sign_reference : (condition, slice), optional
        Flip each component so its mean projection of ``condition`` over the time ``slice`` is >= 0.

    Returns
    -------
    explained : np.ndarray
        Explained-variance ratio of the first ``n_components``.
    proj : dict
        {condition: array (time x n_components)}.
    """
    from sklearn.decomposition import PCA

    keys = list(traces) if fit_on is None else list(fit_on)
    x = np.concatenate([np.asarray(traces[k]).T for k in keys])
    pca = PCA(n_components=n_components, svd_solver="full").fit(x)
    proj = {k: pca.transform(np.asarray(traces[k]).T) for k in keys}
    if sign_reference is not None:
        ref, sl = sign_reference
        for c in range(n_components):
            if proj[ref][sl, c].mean() < 0:
                for k in keys:
                    proj[k][:, c] *= -1
    return pca.explained_variance_ratio_[:n_components], proj
