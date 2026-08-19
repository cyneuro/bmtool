"""Shared fixtures for BMTool unit tests."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# Stub NEURON so modules that import it at the top level can be unit-tested in CI.
if "neuron" not in sys.modules:
    neuron_stub = ModuleType("neuron")
    neuron_stub.h = SimpleNamespace()
    sys.modules["neuron"] = neuron_stub


@pytest.fixture
def rng():
    """Deterministic RNG for tests that exercise stochastic helpers."""
    return np.random.default_rng(42)
