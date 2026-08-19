"""Unit tests for pure math helpers in bmtool.util.util."""

import numpy as np
import pytest

from bmtool.connectors import num_prop as connector_num_prop
from bmtool.util.util import lognormal, num_prop

pytestmark = pytest.mark.unit


class TestUtilNumProp:
    def test_matches_connector_helper(self):
        ratio = [1, 2, 3]
        np.testing.assert_array_equal(num_prop(ratio, 12), connector_num_prop(ratio, 12))

    def test_sums_to_n(self):
        counts = num_prop([2, 2, 1], 10)
        assert int(counts.sum()) == 10


class TestLognormal:
    def test_mean_is_near_requested_value(self, rng):
        samples = lognormal(mean=10.0, stdev=1.0, size=20_000, rng=rng)
        assert samples.mean() == pytest.approx(10.0, rel=0.05)

    def test_scalar_sample_is_positive(self, rng):
        value = lognormal(mean=2.0, stdev=0.5, rng=rng)
        assert np.isscalar(value) or getattr(value, "shape", None) == ()
        assert float(value) > 0
