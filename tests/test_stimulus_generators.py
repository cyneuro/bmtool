"""Unit tests for stimulus firing-rate generators (numpy only)."""

import numpy as np
import pytest

from bmtool.stimulus.generators import (
    get_fr_long,
    get_fr_loop,
    get_fr_short,
    get_stim_cycle,
)

pytestmark = pytest.mark.unit


class TestGetStimCycle:
    def test_basic_cycle_count(self):
        t_cycle, n_cycle = get_stim_cycle(on_time=1.0, off_time=1.0, t_start=0.0, t_stop=10.0)
        assert t_cycle == pytest.approx(2.0)
        assert n_cycle == 5

    def test_respects_t_start(self):
        t_cycle, n_cycle = get_stim_cycle(on_time=2.0, off_time=2.0, t_start=4.0, t_stop=12.0)
        assert t_cycle == pytest.approx(4.0)
        assert n_cycle == 2


def _assert_trace_schema(params, n_assemblies, t_stop=10.0):
    assert len(params) == n_assemblies
    for trace in params:
        assert set(trace) == {"firing_rate", "times"}
        assert len(trace["firing_rate"]) == len(trace["times"])
        assert trace["times"][0] == pytest.approx(0.0)
        assert trace["times"][-1] == pytest.approx(t_stop)


class TestGetFrShort:
    def test_returns_one_trace_per_assembly(self):
        params = get_fr_short(
            n_assemblies=3,
            firing_rate=(1.0, 20.0, 0.0),
            on_time=1.0,
            off_time=0.5,
            t_start=0.0,
            t_stop=10.0,
        )
        _assert_trace_schema(params, 3)

    def test_pads_short_firing_rate_tuple(self):
        params = get_fr_short(n_assemblies=1, firing_rate=(5.0,), t_stop=3.0)
        assert len(params) == 1
        assert all(rate >= 0 for rate in params[0]["firing_rate"])

    def test_empty_assembly_index_still_returns_all_traces(self):
        params = get_fr_short(
            n_assemblies=2,
            firing_rate=(1.0, 10.0, 0.0),
            assembly_index=[],
            t_stop=3.0,
        )
        assert len(params) == 2

    def test_selected_assembly_reaches_burst_rate(self):
        burst_fr = 40.0
        params = get_fr_short(
            n_assemblies=2,
            firing_rate=(1.0, burst_fr, 0.0),
            on_time=1.0,
            off_time=0.5,
            t_start=0.0,
            t_stop=4.0,
            assembly_index=[0],
        )
        assert burst_fr in params[0]["firing_rate"]
        assert burst_fr not in params[1]["firing_rate"]


class TestGetFrLong:
    def test_one_active_assembly_per_cycle(self):
        burst_fr = 25.0
        params = get_fr_long(
            n_assemblies=2,
            firing_rate=(2.0, burst_fr, 0.0),
            on_time=1.0,
            off_time=0.0,
            t_start=0.0,
            t_stop=2.0,
            n_cycles=2,
        )
        assert burst_fr in params[0]["firing_rate"]
        assert burst_fr in params[1]["firing_rate"]
        assert len(params) == 2

    def test_times_are_monotonic(self):
        params = get_fr_long(
            n_assemblies=1,
            firing_rate=(1.0, 10.0, 0.0),
            t_start=0.0,
            t_stop=5.0,
        )
        times = np.asarray(params[0]["times"])
        assert np.all(np.diff(times) >= 0)


class TestGetFrLoop:
    def test_rejects_mismatched_firing_rate_length(self):
        with pytest.raises(ValueError, match="firing_rate"):
            get_fr_loop(n_assemblies=1, firing_rate=(1.0, 2.0), on_times=(1.0,))

    def test_returns_requested_number_of_assemblies(self):
        params = get_fr_loop(
            n_assemblies=3,
            firing_rate=(1.0, 10.0, 0.0),
            on_times=(1.0,),
            off_time=0.5,
            t_start=0.0,
            t_stop=6.0,
        )
        assert len(params) == 3
        for trace in params:
            assert "firing_rate" in trace
            assert "times" in trace
            assert len(trace["firing_rate"]) == len(trace["times"])
