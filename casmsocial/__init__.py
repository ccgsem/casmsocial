"""Top-level package for casmsocial.

Keep package import lightweight so downstream code and tests can safely do
``import casmsocial`` without triggering model registration or MPI-heavy
imports.

Note: Heat risk models have been moved to the ``wake-county-heat-risk`` package.
"""

from casmsocial.sim_time import SimTime

__all__ = ["CitySocialModel", "SimTime"]


def __getattr__(name: str):
    """Lazily load heavier public exports on first access."""
    if name == "CitySocialModel":
        from casmsocial.citysim.citysocialmodel import CitySocialModel

        return CitySocialModel
    raise AttributeError(name)


def __dir__() -> list[str]:
    """Expose lazy public exports to introspection tools."""
    return sorted(set(globals()) | set(__all__))
