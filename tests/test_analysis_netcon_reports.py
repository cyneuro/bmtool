import numpy as np
import pytest

from bmtool.analysis.netcon_reports import (
    NetconReport,
    relative_to_initial,
    sample_index,
    weight_change_by_group,
)


def test_sample_index_scalar_and_array():
    t = np.array([0.0, 10.0, 20.0, 30.0])
    assert sample_index(t, 10.0) == 1
    assert sample_index(t, 10.5) == 2
    assert sample_index(t, 99.0) == 3
    np.testing.assert_array_equal(sample_index(t, [-1.0, 20.0, 31.0]), [0, 2, 3])


def test_relative_to_initial_zero_handling():
    d = np.array([[2.0, 0.0], [4.0, 3.0]])
    np.testing.assert_allclose(relative_to_initial(d), [[1.0, 0.0], [2.0, 3.0]])
    r = relative_to_initial(d, zero_initial="nan")
    assert np.isnan(r[:, 1]).all() and r[1, 0] == 2.0
    with pytest.raises(ValueError):
        relative_to_initial(d, zero_initial="x")


def test_weight_change_by_group_uses_first_sample_at_or_after_mark():
    times = np.array([0.0, 100.0, 200.0])
    data = np.array([[1.0, 1.0], [1.5, 1.0], [2.0, 0.5]])
    rep = NetconReport(times, data, np.array([0, 1]), np.array([5, 6]))
    df = weight_change_by_group(rep, np.array(["a", "b"]), {"mid": 50.0})
    a = df[df.group == "a"].iloc[0]
    assert a.time_ms == 100.0 and a.mean_change == pytest.approx(0.5)
