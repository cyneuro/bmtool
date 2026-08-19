"""Unit tests for connector utilities that do not require NEURON or BMTK."""

import time

import numpy as np
import pytest

from bmtool.connectors import (
    DELAY_LOWBOUND,
    DELAY_UPBOUND,
    AbstractConnector,
    GaussianDropoff,
    NormalizedReciprocalRate,
    Timer,
    UniformInRange,
    cylindrical_dist_z,
    decision,
    decisions,
    euclid_dist,
    gaussian,
    is_same_pop,
    num_prop,
    pr_2_rho,
    rho_2_pr,
    spherical_dist,
    syn_const_delay,
    syn_dist_delay_feng,
    syn_section_PN,
    syn_uniform_delay_section,
)

pytestmark = pytest.mark.unit


class TestNumProp:
    def test_equal_ratio_sums_to_n(self):
        counts = num_prop([1, 1, 1], 9)
        assert counts.tolist() == [3, 3, 3]
        assert int(counts.sum()) == 9

    def test_uneven_ratio(self):
        counts = num_prop([1, 3], 8)
        assert counts.tolist() == [2, 6]
        assert int(counts.sum()) == 8

    def test_preserves_2d_shape(self):
        counts = num_prop([[1, 1], [2, 2]], 12)
        assert counts.shape == (2, 2)
        assert int(counts.sum()) == 12

    def test_zero_n(self):
        counts = num_prop([1, 2, 3], 0)
        assert counts.tolist() == [0, 0, 0]


class TestDecision:
    def test_probability_zero_is_always_false(self, rng):
        assert not bool(decision(0.0, rng=rng))
        result = decision(0.0, size=100, rng=rng)
        assert not np.any(result)

    def test_probability_one_is_always_true(self, rng):
        result = decision(1.0, size=50, rng=rng)
        assert np.all(result)

    def test_seeded_decisions_are_reproducible(self):
        a = decisions([0.2, 0.5, 0.8], rng=np.random.default_rng(7))
        b = decisions([0.2, 0.5, 0.8], rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_decisions_shape_matches_input(self, rng):
        prob = np.full((3, 4), 0.5)
        result = decisions(prob, rng=rng)
        assert result.shape == (3, 4)
        assert result.dtype == bool


class TestDistances:
    def test_euclid_dist_3_4_5(self):
        assert euclid_dist([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_euclid_dist_identical_points(self):
        assert euclid_dist([1.5, -2.0, 3.0], [1.5, -2.0, 3.0]) == pytest.approx(0.0)

    def test_spherical_dist_uses_positions(self):
        node1 = {"positions": np.array([0.0, 0.0, 0.0])}
        node2 = {"positions": np.array([0.0, 3.0, 4.0])}
        assert spherical_dist(node1, node2) == pytest.approx(5.0)

    def test_cylindrical_dist_ignores_z(self):
        node1 = {"positions": np.array([0.0, 0.0, 100.0])}
        node2 = {"positions": np.array([3.0, 4.0, -50.0])}
        assert cylindrical_dist_z(node1, node2) == pytest.approx(5.0)


class TestGaussian:
    def test_peak_at_mean(self):
        x = np.array([-1.0, 0.0, 1.0])
        values = gaussian(x, mean=0.0, stdev=1.0, pmax=1.0)
        assert values[1] == pytest.approx(1.0)
        assert values[1] > values[0]
        assert values[1] > values[2]

    def test_symmetric_around_mean(self):
        left = gaussian(-2.0, mean=0.0, stdev=1.0)
        right = gaussian(2.0, mean=0.0, stdev=1.0)
        assert left == pytest.approx(right)


class TestUniformInRange:
    def test_probability_inside_range(self):
        fn = UniformInRange(p=0.3, min_dist=10.0, max_dist=50.0)
        assert fn(25.0) == pytest.approx(0.3)
        assert fn.probability(25.0) == pytest.approx(0.3)

    def test_probability_outside_range_is_zero(self):
        fn = UniformInRange(p=0.3, min_dist=10.0, max_dist=50.0)
        assert fn(5.0) == 0.0
        assert fn(80.0) == 0.0

    def test_decisions_respect_distance_mask(self, rng):
        fn = UniformInRange(p=1.0, min_dist=1.0, max_dist=2.0)
        dist = np.array([0.0, 1.5, 3.0])
        result = fn.decisions(dist, rng=rng)
        np.testing.assert_array_equal(result, [False, True, False])

    def test_rejects_invalid_probability(self):
        with pytest.raises(AssertionError):
            UniformInRange(p=1.5)

    def test_rejects_invalid_distance_bounds(self):
        with pytest.raises(AssertionError):
            UniformInRange(p=0.2, min_dist=10.0, max_dist=1.0)


class TestGaussianDropoff:
    def test_probability_peaks_at_mean(self):
        fn = GaussianDropoff(mean=0.0, stdev=10.0, pmax=0.5, max_dist=100.0)
        peak = fn(0.0)
        nearby = fn(10.0)
        far = fn(40.0)
        assert peak == pytest.approx(0.5)
        assert peak > nearby > far

    def test_zero_outside_max_dist(self):
        fn = GaussianDropoff(mean=0.0, stdev=10.0, pmax=0.8, max_dist=20.0)
        assert fn(25.0) == 0.0

    def test_cylindrical_vs_spherical_pmax_from_ptotal(self):
        spherical = GaussianDropoff(
            mean=0.0,
            stdev=20.0,
            ptotal=0.2,
            ptotal_dist_range=(0.0, 50.0),
            dist_type="spherical",
        )
        cylindrical = GaussianDropoff(
            mean=0.0,
            stdev=20.0,
            ptotal=0.2,
            ptotal_dist_range=(0.0, 50.0),
            dist_type="cylindrical",
        )
        assert spherical.pmax > 0
        assert cylindrical.pmax > 0
        assert spherical.pmax != pytest.approx(cylindrical.pmax)

    def test_unknown_dist_type_defaults_to_spherical(self):
        fn = GaussianDropoff(dist_type="not-a-real-metric")
        assert fn.dist_type == "spherical"


class TestCorrelationHelpers:
    def test_rho_and_pr_are_inverses(self):
        p0, p1, pr = 0.3, 0.4, 0.18
        rho = pr_2_rho(p0, p1, pr)
        recovered = rho_2_pr(p0, p1, rho)
        assert recovered == pytest.approx(pr)

    def test_independent_connections_have_zero_rho(self):
        p0, p1 = 0.25, 0.4
        rho = pr_2_rho(p0, p1, p0 * p1)
        assert rho == pytest.approx(0.0)

    def test_pr_2_rho_rejects_invalid_probability(self):
        with pytest.raises(AssertionError):
            pr_2_rho(0.0, 0.5, 0.0)

    def test_rho_2_pr_clips_out_of_range_values(self, capsys):
        p0, p1 = 0.2, 0.3
        pr = rho_2_pr(p0, p1, rho=5.0)
        captured = capsys.readouterr()
        assert "rho changed" in captured.out
        assert 0.0 <= pr <= min(p0, p1)
        assert pr >= p0 + p1 - 1


class TestNormalizedReciprocalRate:
    def test_constant_nrr_scales_joint_probability(self):
        nrr = NormalizedReciprocalRate(NRR=2.0)
        assert nrr(10.0, 0.2, 0.3) == pytest.approx(0.12)

    def test_callable_nrr(self):
        nrr = NormalizedReciprocalRate(NRR=lambda dist: np.where(dist < 5, 2.0, 1.0))
        np.testing.assert_allclose(nrr.probability(np.array([1.0, 10.0]), 0.5, 0.5), [0.5, 0.25])

    def test_decisions_with_condition(self, rng):
        nrr = NormalizedReciprocalRate(NRR=1.0)
        dist = np.zeros(8)
        cond = (0, np.array([True, True, True, True, False, False, False, False]))
        result = nrr.decisions(dist, 1.0, 1.0, cond=cond, rng=rng)
        np.testing.assert_array_equal(result[:4], [True, True, True, True])
        np.testing.assert_array_equal(result[4:], [False, False, False, False])


class TestIsSamePop:
    def test_quick_check_matches_on_filters(self):
        source = _FakeNodePool("net", [0, 1], props={"pop": "A"})
        target = _FakeNodePool("net", [0, 1], props={"pop": "A"})
        assert is_same_pop(source, target, quick=True)

    def test_quick_check_differs_on_filters(self):
        source = _FakeNodePool("net", [0, 1], props={"pop": "A"})
        target = _FakeNodePool("net", [0, 1], props={"pop": "B"})
        assert not is_same_pop(source, target, quick=True)

    def test_strict_check_compares_node_ids(self):
        source = _FakeNodePool("net", [0, 1, 2])
        same = _FakeNodePool("net", [0, 1, 2])
        different = _FakeNodePool("net", [0, 1, 3])
        assert is_same_pop(source, same, quick=False)
        assert not is_same_pop(source, different, quick=False)


class TestTimerAndHelpers:
    def test_timer_elapsed_is_positive(self):
        timer = Timer(unit="ms")
        time.sleep(0.01)
        elapsed = timer.end()
        assert elapsed > 0
        assert timer.unit == "ms"

    def test_timer_default_unit_is_seconds(self):
        timer = Timer(unit="not-a-unit")
        assert timer.unit == "sec"

    def test_constant_function(self):
        fn = AbstractConnector.constant_function(7)
        assert fn() == 7
        assert fn("ignored") == 7


class TestSynapseDelays:
    def test_const_delay_without_fluctuation(self, rng):
        delay = syn_const_delay(
            dist=100.0,
            min_delay=0.8,
            velocity=1000.0,
            fluc_stdev=0.0,
            delay_bound=(0.2, 2.0),
            rng=rng,
        )
        assert delay == pytest.approx(0.9)

    def test_const_delay_is_clipped_to_bounds(self, rng):
        delay = syn_const_delay(
            dist=1e6,
            min_delay=0.8,
            velocity=1.0,
            fluc_stdev=0.0,
            delay_bound=(0.2, 2.0),
            rng=rng,
        )
        assert delay == pytest.approx(2.0)

    def test_dist_delay_uses_source_target_positions(self, rng):
        source = {"positions": np.array([0.0, 0.0, 0.0])}
        target = {"positions": np.array([0.0, 0.0, 100.0])}
        delay = syn_dist_delay_feng(
            source,
            target,
            min_delay=0.8,
            velocity=1000.0,
            fluc_stdev=0.0,
            delay_bound=(0.2, 2.0),
            rng=rng,
        )
        assert delay == pytest.approx(0.9)

    def test_section_pn_follows_probability(self, rng):
        sec_id, sec_x = syn_section_PN(
            None, None, p=1.0, sec_id=(1, 2), sec_x=(0.4, 0.6), rng=rng
        )
        assert (sec_id, sec_x) == (1, 0.4)

        sec_id, sec_x = syn_section_PN(
            None, None, p=0.0, sec_id=(1, 2), sec_x=(0.4, 0.6), rng=rng
        )
        assert (sec_id, sec_x) == (2, 0.6)

    def test_uniform_delay_stays_in_bounds(self, rng):
        delays = [syn_uniform_delay_section(None, None, rng=rng) for _ in range(50)]
        assert all(DELAY_LOWBOUND <= d <= DELAY_UPBOUND for d in delays)


class _FakeNode:
    def __init__(self, node_id):
        self.node_id = node_id


class _FakeNodePool:
    def __init__(self, network_name, node_ids, props=None):
        self.network_name = network_name
        self._NodePool__properties = props or {}
        self._node_ids = list(node_ids)

    def __len__(self):
        return len(self._node_ids)

    def __iter__(self):
        return iter(_FakeNode(node_id) for node_id in self._node_ids)

