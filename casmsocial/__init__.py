"""Top-level package for casmsocial.

The imports below surface a concise public API so downstream users can do
``from casmsocial import SimTime`` without needing to know the internal module
layout.

Note: Heat risk models have been moved to the ``wake-county-heat-risk`` package.
"""

# Re-export the most commonly used entry points.
from casmsocial.citysim.citysocialmodel import CitySocialModel
from casmsocial.sim_time import SimTime

__all__ = [
    "CitySocialModel",
    "SimTime",
]
