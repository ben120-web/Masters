"""Constrained model-predictive control for aircraft pitch dynamics."""

from .controller import (
    AIRCRAFT_A,
    AIRCRAFT_B,
    AircraftConstraints,
    AircraftPitchMPC,
    MPCSolution,
    SimulationResult,
    default_controller,
)

__all__ = [
    "AIRCRAFT_A",
    "AIRCRAFT_B",
    "AircraftConstraints",
    "AircraftPitchMPC",
    "MPCSolution",
    "SimulationResult",
    "default_controller",
]
