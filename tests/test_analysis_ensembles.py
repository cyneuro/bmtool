"""Unit tests for bmtool.analysis.ensembles and the netcon / ISI helpers."""

import h5py
import numpy as np
import pytest

from bmtool.analysis import ensembles as en
from bmtool.analysis import netcon_reports as nr
from bmtool.analysis import spikes as sp

pytestmark = pytest.mark.unit


class TestSimilarity:
    def test_cosine_matches_numpy(self, rng):
        x = rng.normal(size=(12, 40))
        xn = x / np.linalg.norm(x, axis=1, keepdims=True)
        np.testing.assert_allclose(en.pairwise_cosine_similarity(x), xn @ xn.T, atol=1e-12)

    def test_per_node_excludes_diagonal(self):
        s = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.0], [0.4, 0.0, 1.0]])
        np.testing.assert_allclose(en.per_node_similarity(s, "mean"), [0.3, 0.1, 0.2])
        np.testing.assert_allclose(en.per_node_similarity(s, "median"), [0.3, 0.1, 0.2])

    def test_bootstrap_ci_brackets_statistic(self, rng):
        x = rng.normal(1.0, 1.0, 400)
        m, lo, hi = en.bootstrap_ci(x, np.median, n_resamples=500, seed=1)
        assert lo < m < hi and hi - lo < 0.5


class TestPCA:
    def test_matches_svd_and_sign(self, rng):
        traces = {k: rng.normal(size=(20, 15)) for k in ("a", "b")}
        traces["a"][:5, 5:10] += 3
        ev, proj = en.trajectory_pca(traces, 3, sign_reference=("a", slice(5, 10)))
        x = np.concatenate([traces[k].T for k in ("a", "b")])
        mu = x.mean(0)
        _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
        np.testing.assert_allclose(ev, (s ** 2 / (s ** 2).sum())[:3])
        ref = (traces["a"].T - mu) @ vt[:3].T
        for c in range(3):
            sign = 1 if ref[5:10, c].mean() >= 0 else -1
            np.testing.assert_allclose(proj["a"][:, c], sign * ref[:, c], atol=1e-9)
            assert proj["a"][5:10, c].mean() >= 0


class TestISI:
    def test_window_isis_and_cv2(self):
        t = np.array([0.0, 10, 20, 30, 100, 150])
        isis = sp.window_isis(t, [(0, 35), (95, 200)])
        assert [list(i) for i in isis] == [[10, 10, 10], [50]]
        assert sp.isi_local_variation(isis) == 0.0  # regular within the first window; the second has one ISI

    def test_short_isi_clusters(self):
        pairs, in_runs = sp.short_isi_clusters([np.array([5, 5, 50, 5, 5, 5, 50])], short=10)
        assert pairs == 3 and in_runs == 3 + 4


class TestNetcon:
    def test_read_weight_change_and_bands(self, tmp_path):
        p = tmp_path / "w.h5"
        data = np.array([[1.0, 2.0, 1.0, 0.0], [1.2, 2.0, 0.8, 1.0], [1.5, 1.0, 0.5, 1.0]])
        with h5py.File(p, "w") as f:
            g = f.create_group("report/LA")
            g.create_dataset("data", data=data)
            m = g.create_group("mapping")
            m.create_dataset("time", data=[0.0, 30.0, 10.0])
            m.create_dataset("trg_ids", data=[0, 1, 2, 3]); m.create_dataset("src_ids", data=[9, 9, 9, 9])
        assert nr.read_netcon_report(tmp_path / "missing.h5", "LA") is None
        rep = nr.read_netcon_report(p, "LA")
        np.testing.assert_allclose(rep.times, [0, 10, 20])
        out = nr.weight_change_by_group(rep, np.array(["a", "a", "b", "b"]), {"mid": 10.0, "end": 25.0})
        row = out[(out.group == "a") & (out.mark == "mid")].iloc[0]
        assert row.mean_change == pytest.approx(0.1) and row.frac_ltp == pytest.approx(0.5)
        end_b = out[(out.group == "b") & (out.mark == "end")].iloc[0]  # W0 = 0 synapse ignored in the mean
        assert end_b.mean_change == pytest.approx(-0.5) and end_b.frac_ltd == pytest.approx(0.5)
        bf = nr.band_fractions(data, rep.times, np.array([0, 1]), [(0, 15)], low=1.1, high=1.9)
        assert bf["frac_between"] == pytest.approx(0.25) and bf["frac_above"] == pytest.approx(0.5)
        assert nr.band_fractions(data, rep.times, np.array([0]), [(100, 200)], 0, 1) is None
