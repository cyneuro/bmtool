"""Stimulus helpers.

Submodules and ``StimulusBuilder`` are imported lazily so generator utilities
can be used without BMTK installed.
"""

__all__ = ["StimulusBuilder", "assemblies", "generators"]


def __getattr__(name):
    if name == "StimulusBuilder":
        from .core import StimulusBuilder

        return StimulusBuilder
    if name in {"assemblies", "generators"}:
        from importlib import import_module

        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
