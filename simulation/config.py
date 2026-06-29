"""
Configuration of the reference orbit used in the project.

This module loads the classical orbital elements and simulation settings from
YAML files. The orbit is later propagated using poliastro.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import astropy.units as u
import yaml
from astropy.time import Time

DEFAULT_SETTINGS_DIR = Path(__file__).resolve().parent / "settings"
DEFAULT_ORBIT_CONFIG_PATH = DEFAULT_SETTINGS_DIR / "orbit.yaml"


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


def load_yaml_file(path: Path) -> dict[str, Any]:
    """
    Load a YAML file and return its top-level mapping.
    """

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a top-level mapping: {path}")

    return data


def _get_section(data: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a named YAML section as a mapping."""

    section = data.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{section_name}' section in YAML config.")

    return section


def _get_float(section: dict[str, Any], key: str) -> float:
    """Return a required numeric YAML value as float."""

    value = section.get(key)

    if not isinstance(value, int | float):
        raise ValueError(f"Missing or invalid numeric value: {key}")

    return float(value)


def _get_string(section: dict[str, Any], key: str) -> str:
    """Return a required string YAML value."""

    value = section.get(key)

    if not isinstance(value, str):
        raise ValueError(f"Missing or invalid string value: {key}")

    return value


def create_orbit_from_yaml(path: Path = DEFAULT_ORBIT_CONFIG_PATH) -> ClassicalOrbitalElements:
    """
    Create classical orbital elements from a YAML configuration file.
    """

    data = load_yaml_file(path)
    orbit = _get_section(data, "orbit")

    return ClassicalOrbitalElements(
        semi_major_axis=_get_float(orbit, "semi_major_axis_m") * u.m,
        eccentricity=_get_float(orbit, "eccentricity") * u.one,
        inclination=np.deg2rad(_get_float(orbit, "inclination_deg")) * u.rad,
        raan=np.deg2rad(_get_float(orbit, "raan_deg")) * u.rad,
        argument_of_perigee=np.deg2rad(_get_float(orbit, "argument_of_perigee_deg")) * u.rad,
        true_anomaly=np.deg2rad(_get_float(orbit, "true_anomaly_deg")) * u.rad,
        epoch=Time(_get_string(orbit, "epoch_utc"), scale="utc"),
    )


def create_simulation_config_from_yaml(path: Path = DEFAULT_ORBIT_CONFIG_PATH) -> SimulationConfig:
    """
    Create simulation time settings from a YAML configuration file.
    """

    data = load_yaml_file(path)
    simulation = _get_section(data, "simulation")

    return SimulationConfig(
        duration_s=_get_float(simulation, "duration_s"),
        time_step_s=_get_float(simulation, "time_step_s"),
    )


def create_default_orbit() -> ClassicalOrbitalElements:
    """
    Create the default LEO orbit from the YAML configuration.

    Returns
    -------
    ClassicalOrbitalElements
        Default orbit configuration.
    """

    return create_orbit_from_yaml(DEFAULT_ORBIT_CONFIG_PATH)


def create_default_simulation_config() -> SimulationConfig:
    """
    Create default simulation settings from the YAML configuration.

    Returns
    -------
    SimulationConfig
        Simulation duration and sampling time.
    """

    return create_simulation_config_from_yaml(DEFAULT_ORBIT_CONFIG_PATH)
