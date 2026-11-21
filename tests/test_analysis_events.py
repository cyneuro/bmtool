"""Unit tests for bmtool.analysis.events (event-aligned spike analysis)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bmtool.analysis import events as ev

pytestmark = pytest.mark.unit


def df(ids, ts):
    return pd.DataFrame({"node_ids": ids, "timestamps": ts})


def poisson_df(rng, n_nodes, t_stop, rate_hz, extra=None):
    ids, ts = [], []
    for n in range(n_nodes):
        t = np.sort(rng.uniform(0, t_stop, rng.poisson(rate_hz * t_stop / 1000)))
        if extra is not None:
            t = np.sort(np.concatenate([t, extra(n, rng)]))
        ids.append(np.full(len(t), n))
        ts.append(t)
    return df(np.concatenate(ids), np.concatenate(ts))


class TestCounts:
    def test_half_open_windows_and_row_order(self):
        d = df([2, 2, 5, 5, 5], [100.0, 150.0, 99.9, 100.0, 200.0])
        c = ev.count_spikes_in_windows(d, [100.0], (0.0, 100.0), node_ids=[5, 2, 7])
        assert c.dims == ("node_id", "event")
        np.testing.assert_array_equal(c["node_id"], [5, 2, 7])
        np.testing.assert_array_equal(c.values[:, 0], [1, 2, 0])  # 99.9 and 200.0 excluded

    def test_matches_brute_force(self, rng):
        d = poisson_df(rng, 20, 5000, 10)
        on = rng.uniform(500, 4000, 7)
        c = ev.count_spikes_in_windows(d, on, (-100, 250), node_ids=np.arange(20)).values
        ref = np.array(
            [
                [
                    np.sum(
                        (d.timestamps[d.node_ids == n] >= o - 100)
                        & (d.timestamps[d.node_ids == n] < o + 250)
                    )
                    for o in on
                ]
                for n in range(20)
            ]
        )
        np.testing.assert_array_equal(c, ref)


class TestBinning:
    def test_bin_spikes_matches_histogram(self, rng):
        d = poisson_df(rng, 5, 2000, 20)
        b = ev.bin_spikes(d, 100.0, 0.0, 2000.0, node_ids=np.arange(5), rate=True)
        for n in range(5):
            h, _ = np.histogram(d.timestamps[d.node_ids == n], np.arange(0, 2100, 100.0))
            np.testing.assert_allclose(b.values[n], h / 0.1)
        assert b.attrs["units"] == "Hz"

    def test_event_aligned_shape_and_values(self, rng):
        d = poisson_df(rng, 3, 3000, 30)
        b = ev.bin_spikes(d, 50.0, 0.0, 3000.0, node_ids=np.arange(3))
        a = ev.event_aligned(b, [1000.0, 2000.0], (-100.0, 200.0))
        assert a.shape == (3, 2, 6)
        np.testing.assert_array_equal(a.values[:, 1, :], b.values[:, 38:44])
        np.testing.assert_allclose(a["lag"], [-100, -50, 0, 50, 100, 150])

    def test_peth_population_and_per_node(self):
        d = df([0, 0, 1], [1010.0, 1060.0, 2010.0])
        p = ev.peth(d, [1000.0, 2000.0], (0.0, 100.0), 50.0, node_ids=[0, 1, 2])
        # bin 0: 2 spikes, bin 1: 1 spike; / (2 events * 0.05 s) / 3 nodes
        np.testing.assert_allclose(p.values, [2 / 0.1 / 3, 1 / 0.1 / 3])
        pn = ev.peth(d, [1000.0, 2000.0], (0.0, 100.0), 50.0, node_ids=[0, 1, 2], per_node=True)
        np.testing.assert_allclose(pn.values[0], [10.0, 10.0])
        np.testing.assert_allclose(pn.mean("node_id").values, p.values)

    def test_baseline_zscore_floor_and_nan(self):
        import xarray as xr

        x = xr.DataArray(np.array([[1.0, 1.0, 5.0], [0.0, 2.0, 4.0]]), dims=("node_id", "time"))
        z = ev.baseline_zscore(x, [True, True, False])
        assert np.isnan(z.values[0]).all()  # zero baseline SD
        np.testing.assert_allclose(z.values[1], [-1, 1, 3])
        zf = ev.baseline_zscore(x, [True, True, False], sd_floor=2.0)
        np.testing.assert_allclose(zf.values[0], [0, 0, 2])


class TestClassify:
    def test_poisson_up_down_none(self):
        cls, p = ev.classify_response(
            [40, 2, 5, 0], [10, 30, 5, 0], resp_duration=1.0, base_duration=1.0
        )
        assert list(cls) == ["Up", "Down", "None", "None"]
        assert np.isnan(p[3]) and p[0] < 1e-3

    def test_poisson_uses_durations(self):
        # 10 spikes in 1 s vs 10 spikes in 10 s: clearly Up
        cls, _ = ev.classify_response([10], [10], resp_duration=1.0, base_duration=10.0)
        assert cls[0] == "Up"

    def test_ranksum_two_d(self, rng):
        resp = rng.poisson(5, (2, 30))
        base = rng.poisson(5, (2, 30))
        resp[0] += 5
        cls, _ = ev.classify_response(resp, base, test="ranksum")
        assert cls[0] == "Up"

    def test_two_d_poisson_matches_sums(self, rng):
        resp, base = rng.poisson(2, (5, 10)), rng.poisson(1, (5, 20))
        c2, p2 = ev.classify_response(resp, base, test="poisson")
        c1, p1 = ev.classify_response(resp.sum(1), base.sum(1), resp_duration=10, base_duration=20)
        assert list(c1) == list(c2)
        np.testing.assert_allclose(p1, p2)


class TestPopulation:
    def test_percent_and_rate_change(self):
        # node 0: 2 spikes in each 100 ms response window, 1 spike in each 100 ms baseline window
        d = df([0] * 6, [1010, 1020, 950, 2010, 2020, 1950])
        on = [1000.0, 2000.0]
        assert ev.population_percent_change(d, on, (0, 100), (-100, 0), [0]) == pytest.approx(100.0)
        assert ev.population_rate_change(d, on, (0, 100), (-100, 0), [0]) == pytest.approx(10.0)

    def test_response_z_positive_for_driven_population(self, rng):
        d = poisson_df(
            rng,
            30,
            20000,
            5,
            extra=lambda n, r: np.concatenate(
                [o + r.uniform(0, 100, 3) for o in np.arange(2000, 19000, 2000)]
            ),
        )
        z = ev.population_response_z(
            d, np.arange(2000, 19000, 2000.0), (0, 100), (-500, 0), 50.0, np.arange(30)
        )
        assert z > 3

    def test_response_auc(self):
        import xarray as xr

        x = xr.DataArray(
            np.array([[1.0, 2.0, 3.0, 4.0]]),
            dims=("node_id", "lag"),
            coords={"lag": [0, 10, 20, 30]},
        )
        np.testing.assert_allclose(ev.response_auc(x, (10, 30)).values, [2.5 * 20])


class TestRepa:
    def test_matches_reference_loop(self, rng):
        on = np.arange(1000.0, 1000.0 + 24 * 1500, 1500)
        d = poisson_df(
            rng,
            8,
            on[-1] + 2000,
            8,
            extra=lambda n, r: np.concatenate(
                [o + r.uniform(0, 200, 2 if (k >= 8 and n < 4) else 0) for k, o in enumerate(on)]
            ),
        )
        z = ev.repa_zscores(d, on, baseline_events=range(8), node_ids=np.arange(8))
        # reference: the per-node loop of the original notebook helper
        for n in range(8):
            t = d.timestamps[d.node_ids == n].values
            base = [
                np.sum((t >= b) & (t < b + 10)) for o in on[:8] for b in np.arange(o - 200, o, 10)
            ]
            mu, sd = np.mean(base), np.std(base)
            ref = [
                np.mean(
                    [
                        (np.sum((t >= o + b) & (t < o + b + 10)) - mu) / sd if sd > 0 else 0.0
                        for b in np.arange(0, 200, 10)
                    ]
                )
                for o in on
            ]
            np.testing.assert_allclose(z.values[n], ref, atol=1e-12)
        pl = ev.repa_plastic_cells(z, range(8), range(8, 24))
        assert pl.loc[pl.node_id < 4, "is_plastic"].all()
        md = ev.repa_max_block_difference(z, range(8), range(8, 24))
        assert (md.n_blocks == 4).all()
        driven = md.loc[md.node_id < 4, "max_z_difference"]
        assert driven.min() > md.loc[md.node_id >= 4, "max_z_difference"].max()


def test_baseline_zscore_sd_offset():
    x = xr.DataArray(np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 2.0]]), dims=("node_id", "time"))
    z = ev.baseline_zscore(x, np.array([True, True, False]), sd_offset=1.0)
    np.testing.assert_allclose(z.values[0], (x.values[0] - 2.0) / (1.0 + 1.0))
    np.testing.assert_allclose(z.values[1], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        ev.baseline_zscore(x, [0, 1], sd_floor=1.0, sd_offset=1.0)
