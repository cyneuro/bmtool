"""Smoke tests for bmtool.bmplot.events."""

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from bmtool.analysis import events as ev  # noqa: E402
from bmtool.bmplot import events as pe  # noqa: E402

pytestmark = pytest.mark.unit


def test_plots_run(rng):
    d = pd.DataFrame(
        {"node_ids": rng.integers(0, 10, 500), "timestamps": rng.uniform(0, 5000, 500)}
    )
    p = ev.peth(d, [1000.0, 2000.0, 3000.0], (-200.0, 400.0), 50.0, node_ids=np.arange(10))
    ax = pe.plot_peth(p, events=[(0, 200)])
    assert len(ax.lines) == 1
    z = xr.DataArray(
        rng.normal(size=(10, 12)), dims=("node_id", "lag"), coords={"lag": np.arange(12) * 50.0}
    )
    pe.plot_event_heatmap(z, sort_window=(0, 200))
    _, res = pe.plot_paired_bars(
        rng.normal(0, 1, 50), rng.normal(0.5, 1, 50), stat="median", n_resamples=200
    )
    assert set(res) == {"before", "after", "p", "frac_up"}
    pe.plot_node_differences(rng.normal(size=30), groups=["a"] * 15 + ["b"] * 15)
