"""Smoke tests that the package is importable without NEURON or BMTK."""

import pytest

pytestmark = pytest.mark.unit


def test_bmtool_version_is_set():
    import bmtool

    assert isinstance(bmtool.__version__, str)
    assert bmtool.__version__
    assert bmtool.__version__ != "0+unknown"


def test_connectors_import_without_neuron():
    import bmtool.connectors as connectors

    assert callable(connectors.num_prop)
    assert callable(connectors.euclid_dist)


def test_stimulus_generators_import_without_bmtk():
    from bmtool.stimulus import generators

    assert callable(generators.get_stim_cycle)
    assert callable(generators.get_fr_short)


def test_stimulus_builder_import_requires_bmtk():
    import bmtool.stimulus as stimulus

    with pytest.raises(ModuleNotFoundError, match="bmtk"):
        stimulus.StimulusBuilder
