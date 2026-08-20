"""Configuration loading and validation for simulation inputs."""

from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import astropy.units as u
import yaml
from astropy.time import Time

from simulation.helpers import normalize_quaternion
from simulation.types import (
    AttitudeConfig,
    ArrayFloat64,
    ClassicalOrbitalElements,
    MagnetometerConfig,
    SimulationConfig,
)

DEFAULT_SETTINGS_DIR = Path(__file__).resolve().parent.parent / "settings"
DEFAULT_ORBIT_CONFIG_PATH = DEFAULT_SETTINGS_DIR / "orbit.yaml"
DEFAULT_SATELLITE_CONFIG_PATH = DEFAULT_SETTINGS_DIR / "satellite.yaml"


def load_yaml_file(path: Path) -> dict[str, Any]:
    """
    Load a YAML file and return its top-level mapping.
    """

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _get_section(data: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a named YAML section as a mapping."""

    return data[section_name]


def _get_float(section: dict[str, Any], key: str) -> float:
    """Return a required numeric YAML value as float."""

    return float(section[key])


def _get_string(section: dict[str, Any], key: str) -> str:
    """Return a required string YAML value."""

    return section[key]


def _get_optional_int(section: dict[str, Any], key: str) -> int | None:
    """Return an optional integer YAML value."""

    value = section.get(key)

    if value is None:
        return None

    return int(value)


def _get_vector(section: dict[str, Any], key: str) -> ArrayFloat64:
    """Return a required numeric vector YAML value."""

    return np.array(section[key], dtype=float)


def _get_matrix(section: dict[str, Any], key: str) -> ArrayFloat64:
    """Return a required numeric matrix YAML value."""

    return np.array(section[key], dtype=float)


def _validate_attitude_config(config: AttitudeConfig) -> None:
    """Validate attitude dynamics settings."""

    if config.mass_kg <= 0.0:
        raise ValueError("Satellite mass must be positive.")

    if not np.allclose(config.inertia_kg_m2, config.inertia_kg_m2.T):
        raise ValueError("Inertia matrix must be symmetric.")

    if np.any(np.linalg.eigvalsh(config.inertia_kg_m2) <= 0.0):
        raise ValueError("Inertia matrix must be positive definite.")


def _validate_rotation_matrix(matrix: ArrayFloat64, name: str) -> None:
    """Validate a proper 3D rotation matrix."""

    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.allclose(matrix @ matrix.T, np.eye(3, dtype=np.float64), atol=1.0e-9):
        raise ValueError(f"{name} must be orthonormal.")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1.0e-9):
        raise ValueError(f"{name} must have determinant +1.")


def _validate_sensor_axes_matrix(matrix: ArrayFloat64, name: str) -> None:
    """Validate an orthonormal set of sensor sensitive axes."""

    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    if not np.allclose(matrix @ matrix.T, np.eye(3, dtype=np.float64), atol=1.0e-9):
        raise ValueError(f"{name} must be orthonormal.")


def _validate_sensor_positions_matrix(matrix: ArrayFloat64, name: str) -> None:
    """Validate sensor positions in body-frame coordinates."""

    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")


def _validate_magnetometer_config(config: MagnetometerConfig) -> None:
    """Validate magnetometer mounting and noise settings."""

    if config.bias_sensor_t.shape != (3,):
        raise ValueError("Magnetometer bias must have shape (3,).")
    if isinstance(config.noise_std_t, Real):
        if float(config.noise_std_t) < 0.0:
            raise ValueError("Magnetometer noise standard deviation must be nonnegative.")
    else:
        noise_std_t = np.asarray(config.noise_std_t, dtype=np.float64)

        if noise_std_t.shape != (3,):
            raise ValueError("Magnetometer noise standard deviation must be scalar or shape (3,).")
        if np.any(noise_std_t < 0.0):
            raise ValueError("Magnetometer noise standard deviation must be nonnegative.")

    _validate_sensor_axes_matrix(config.sensor_axes_from_body, "sensor_axes_from_body")
    _validate_sensor_positions_matrix(config.positions_body_m, "positions_body_m")


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


def create_attitude_config_from_yaml(path: Path = DEFAULT_SATELLITE_CONFIG_PATH) -> AttitudeConfig:
    """
    Create attitude and satellite settings from a YAML configuration file.
    """

    data = load_yaml_file(path)
    satellite = _get_section(data, "satellite")
    attitude = _get_section(data, "attitude")
    integration = _get_section(attitude, "integration")

    config = AttitudeConfig(
        mass_kg=_get_float(satellite, "mass_kg"),
        inertia_kg_m2=_get_matrix(satellite, "inertia_kg_m2"),
        initial_quaternion_eci_from_body=normalize_quaternion(
            _get_vector(attitude, "initial_quaternion_eci_from_body")
        ),
        initial_omega_body_radps=np.deg2rad(_get_vector(attitude, "initial_omega_body_degps")),
        torque_body_nm=_get_vector(attitude, "torque_body_nm"),
        integration_method=_get_string(integration, "method"),
        rtol=_get_float(integration, "rtol"),
        atol=_get_float(integration, "atol"),
    )
    _validate_attitude_config(config)

    return config


def create_magnetometer_config_from_yaml(
    path: Path = DEFAULT_SATELLITE_CONFIG_PATH,
) -> MagnetometerConfig:
    """
    Create magnetometer mounting and noise settings from a YAML configuration file.
    """

    data = load_yaml_file(path)
    magnetometer = _get_section(data, "magnetometer")

    config = MagnetometerConfig(
        bias_sensor_t=_get_vector(magnetometer, "bias_sensor_uT") * 1.0e-6,
        noise_std_t=_get_float(magnetometer, "noise_std_uT") * 1.0e-6,
        seed=_get_optional_int(magnetometer, "seed"),
        sensor_axes_from_body=_get_matrix(
            magnetometer,
            (
                "sensor_axes_from_body"
                if "sensor_axes_from_body" in magnetometer
                else "rotation_sensor_from_body"
            ),
        ),
        positions_body_m=(
            _get_matrix(magnetometer, "positions_body_m")
            if "positions_body_m" in magnetometer
            else np.zeros((3, 3), dtype=np.float64)
        ),
    )
    _validate_magnetometer_config(config)

    return config


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


def create_default_attitude_config() -> AttitudeConfig:
    """
    Create default attitude settings from the YAML configuration.
    """

    return create_attitude_config_from_yaml(DEFAULT_SATELLITE_CONFIG_PATH)


def create_default_magnetometer_config() -> MagnetometerConfig:
    """
    Create default magnetometer settings from the YAML configuration.
    """

    return create_magnetometer_config_from_yaml(DEFAULT_SATELLITE_CONFIG_PATH)
