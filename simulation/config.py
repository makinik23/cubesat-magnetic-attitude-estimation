"""
Configuration of the reference orbit used in the project.

This module defines the classical orbital elements and simulation settings.
The orbit is later propagated using poliastro.
"""

from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.time import Time


@dataclass(frozen=True)
class ClassicalOrbitalElements:
    """
    Classical Keplerian orbital elements.

    All angular quantities are stored as astropy quantities.
    """

    semi_major_axis: u.Quantity
    eccentricity: u.Quantity
    inclination: u.Quantity
    raan: u.Quantity
    argument_of_perigee: u.Quantity
    true_anomaly: u.Quantity
    epoch: Time


@dataclass(frozen=True)
class SimulationConfig:
    """
    Time configuration of the simulation.
    """

    duration_s: float
    time_step_s: float


def create_default_orbit() -> ClassicalOrbitalElements:
    """
    Create a default LEO orbit for the first project milestone.

    Returns
    -------
    ClassicalOrbitalElements
        Default orbit configuration.
    """

    earth_radius_m = 6378137.0
    altitude_m = 500e3

    return ClassicalOrbitalElements(
        semi_major_axis=(earth_radius_m + altitude_m) * u.m,
        eccentricity=0.001 * u.one,
        inclination=np.deg2rad(97.0) * u.rad,
        raan=np.deg2rad(0.0) * u.rad,
        argument_of_perigee=np.deg2rad(0.0) * u.rad,
        true_anomaly=np.deg2rad(0.0) * u.rad,
        epoch=Time("2026-01-01T12:00:00.000", scale="utc"),
    )


def create_default_simulation_config() -> SimulationConfig:
    """
    Create default simulation settings.

    Returns
    -------
    SimulationConfig
        Simulation duration and sampling time.
    """

    return SimulationConfig(duration_s=6000.0, time_step_s=10.0)
