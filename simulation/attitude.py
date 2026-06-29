"""
Rigid-body attitude propagation and body-frame magnetic-field projection.

The initial attitude is provided as a scalar-first quaternion. The propagated
quaternion maps body-frame vectors into ECI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import solve_ivp

from simulation.config import DEFAULT_SETTINGS_DIR, load_yaml_file

ArrayFloat64 = npt.NDArray[np.float64]
DEFAULT_SATELLITE_CONFIG_PATH = DEFAULT_SETTINGS_DIR / "satellite.yaml"


@dataclass(frozen=True)
class AttitudeConfig:
    """
    Rigid-body attitude propagation settings.

    Parameters
    ----------
    mass_kg:
        Spacecraft mass [kg].
    inertia_kg_m2:
        Spacecraft inertia matrix in body coordinates [kg m^2].
    initial_quaternion_eci_from_body:
        Initial scalar-first quaternion that maps body coordinates into ECI.
    initial_omega_body_radps:
        Initial angular velocity in body coordinates [rad/s].
    torque_body_nm:
        Constant external torque in body coordinates [N m].
    integration_method:
        solve_ivp integration method.
    rtol:
        Relative integration tolerance.
    atol:
        Absolute integration tolerance.
    """

    mass_kg: float
    inertia_kg_m2: ArrayFloat64
    initial_quaternion_eci_from_body: ArrayFloat64
    initial_omega_body_radps: ArrayFloat64
    torque_body_nm: ArrayFloat64
    integration_method: str
    rtol: float
    atol: float


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


def _get_vector(section: dict[str, Any], key: str, length: int = 3) -> ArrayFloat64:
    """Return a required numeric vector YAML value."""

    value = section.get(key)
    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (length,):
        raise ValueError(f"Missing or invalid vector value: {key}")

    return vector


def _get_matrix(section: dict[str, Any], key: str, shape: tuple[int, int] = (3, 3)) -> ArrayFloat64:
    """Return a required numeric matrix YAML value."""

    value = section.get(key)
    matrix = np.asarray(value, dtype=np.float64)

    if matrix.shape != shape:
        raise ValueError(f"Missing or invalid matrix value: {key}")

    return matrix


def _validate_attitude_config(config: AttitudeConfig) -> None:
    """Validate mass and inertia properties used by attitude dynamics."""

    if config.mass_kg <= 0.0:
        raise ValueError("Satellite mass must be positive.")

    if not np.allclose(config.inertia_kg_m2, config.inertia_kg_m2.T):
        raise ValueError("Inertia matrix must be symmetric.")

    if np.any(np.linalg.eigvalsh(config.inertia_kg_m2) <= 0.0):
        raise ValueError("Inertia matrix must be positive definite.")


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
        initial_quaternion_eci_from_body=_normalize_quaternion(
            _get_vector(attitude, "initial_quaternion_eci_from_body", length=4)
        ),
        initial_omega_body_radps=np.deg2rad(_get_vector(attitude, "initial_omega_body_degps")),
        torque_body_nm=_get_vector(attitude, "torque_body_nm"),
        integration_method=_get_string(integration, "method"),
        rtol=_get_float(integration, "rtol"),
        atol=_get_float(integration, "atol"),
    )
    _validate_attitude_config(config)

    return config


def create_default_attitude_config() -> AttitudeConfig:
    """
    Create the default attitude configuration from the YAML file.
    """

    return create_attitude_config_from_yaml(DEFAULT_SATELLITE_CONFIG_PATH)


def _as_vector_array(values: Any, name: str) -> ArrayFloat64:
    """Convert values to a float64 array with shape (N, 3)."""

    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")

    return array


def _normalize_quaternion(quaternion: ArrayFloat64) -> ArrayFloat64:
    """Normalize a scalar-first quaternion."""

    norm = np.linalg.norm(quaternion)

    if norm <= 0.0:
        raise ValueError("Quaternion must have non-zero norm.")

    return quaternion / norm


def quaternion_multiply(left: ArrayFloat64, right: ArrayFloat64) -> ArrayFloat64:
    """Multiply scalar-first quaternions."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right

    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation_matrix(quaternion: ArrayFloat64) -> ArrayFloat64:
    """
    Convert a scalar-first quaternion to an ECI-from-body rotation matrix.
    """

    qw, qx, qy, qz = _normalize_quaternion(quaternion)

    return np.array(
        [
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation_matrix: ArrayFloat64) -> ArrayFloat64:
    """
    Convert a rotation matrix to a scalar-first quaternion.
    """

    matrix = np.asarray(rotation_matrix, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3).")

    trace = np.trace(matrix)

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))

        if diagonal_index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif diagonal_index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )

    quaternion = _normalize_quaternion(quaternion)

    if quaternion[0] < 0.0:
        quaternion = -quaternion

    return quaternion


def rotation_matrix_to_zyx_euler(rotation_matrix: ArrayFloat64) -> ArrayFloat64:
    """
    Convert an ECI-from-body rotation matrix to ZYX yaw, pitch and roll angles.
    """

    matrix = np.asarray(rotation_matrix, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3).")

    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > 1e-12:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])

    return np.array([yaw, pitch, roll], dtype=np.float64)


def rigid_body_derivative(
    quaternion_eci_from_body: ArrayFloat64,
    omega_body_radps: ArrayFloat64,
    inertia_kg_m2: ArrayFloat64,
    torque_body_nm: ArrayFloat64,
) -> tuple[ArrayFloat64, ArrayFloat64]:
    """
    Compute quaternion and angular-velocity derivatives.

    This solves the classical Euler rigid-body equation:
    I * omega_dot + omega x (I * omega) = torque.
    """

    omega_quaternion = np.array([0.0, *omega_body_radps], dtype=np.float64)
    quaternion_dot = 0.5 * quaternion_multiply(quaternion_eci_from_body, omega_quaternion)

    angular_momentum_body = inertia_kg_m2 @ omega_body_radps
    omega_dot_body = np.asarray(
        np.linalg.solve(
            inertia_kg_m2, torque_body_nm - np.cross(omega_body_radps, angular_momentum_body)
        ),
        dtype=np.float64,
    )

    return quaternion_dot, omega_dot_body


def attitude_state_derivative(
    _time_s: float, state: ArrayFloat64, config: AttitudeConfig
) -> ArrayFloat64:
    """Compute the solve_ivp state derivative for attitude propagation."""

    quaternion = _normalize_quaternion(state[:4])
    omega = state[4:]
    quaternion_dot, omega_dot = rigid_body_derivative(
        quaternion, omega, config.inertia_kg_m2, config.torque_body_nm
    )

    return np.concatenate((quaternion_dot, omega_dot))


def propagate_attitude(
    times_s: ArrayFloat64, config: AttitudeConfig
) -> tuple[ArrayFloat64, ArrayFloat64]:
    """
    Propagate torque-free rigid-body attitude over the simulation time grid.
    """

    times_s = np.asarray(times_s, dtype=np.float64)

    if times_s.ndim != 1:
        raise ValueError("times_s must be a one-dimensional array.")

    if len(times_s) == 0:
        raise ValueError("times_s must not be empty.")

    if len(times_s) > 1 and np.any(np.diff(times_s) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")

    initial_state = np.concatenate(
        (
            config.initial_quaternion_eci_from_body,
            np.asarray(config.initial_omega_body_radps, dtype=np.float64),
        )
    )

    if len(times_s) == 1:
        return initial_state[:4].reshape(1, 4), initial_state[4:].reshape(1, 3)

    solution = solve_ivp(
        fun=lambda time_s, state: attitude_state_derivative(time_s, state, config),
        t_span=(float(times_s[0]), float(times_s[-1])),
        y0=initial_state,
        t_eval=times_s,
        method=config.integration_method,
        rtol=config.rtol,
        atol=config.atol,
    )

    if not solution.success:
        raise RuntimeError(f"Attitude integration failed: {solution.message}")

    states = np.asarray(solution.y.T, dtype=np.float64)
    quaternions = states[:, :4]
    quaternion_norms = np.linalg.norm(quaternions, axis=1)

    if np.any(quaternion_norms <= 0.0):
        raise RuntimeError("Attitude integration produced a zero-norm quaternion.")

    quaternions = quaternions / quaternion_norms[:, np.newaxis]
    omegas = states[:, 4:]

    return quaternions, omegas


def append_attitude_columns(df: pd.DataFrame, config: AttitudeConfig | None = None) -> pd.DataFrame:
    """
    Append attitude, body-frame magnetic-field and rotation sanity-check columns.
    """

    if config is None:
        config = create_default_attitude_config()

    required_columns = {"t_s", "Bx_eci_T", "By_eci_T", "Bz_eci_T"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    b_eci_t = _as_vector_array(df[["Bx_eci_T", "By_eci_T", "Bz_eci_T"]].to_numpy(), "b_eci_t")
    times_s = np.asarray(df["t_s"].to_numpy(), dtype=np.float64)

    quaternions, omegas = propagate_attitude(times_s, config)

    rotation_matrices = np.asarray(
        [quaternion_to_rotation_matrix(quaternion) for quaternion in quaternions], dtype=np.float64
    )
    euler_zyx_rad = np.asarray(
        [rotation_matrix_to_zyx_euler(rotation_matrix) for rotation_matrix in rotation_matrices],
        dtype=np.float64,
    )
    rotation_body_from_eci = np.transpose(rotation_matrices, axes=(0, 2, 1))

    b_body_t = np.einsum("nij,nj->ni", rotation_body_from_eci, b_eci_t)
    rt_r_minus_i = np.einsum("nji,njk->nik", rotation_matrices, rotation_matrices) - np.eye(3)
    det_rotation = np.linalg.det(rotation_matrices)

    result = df.copy()
    result["q_eci_from_body_w"] = quaternions[:, 0]
    result["q_eci_from_body_x"] = quaternions[:, 1]
    result["q_eci_from_body_y"] = quaternions[:, 2]
    result["q_eci_from_body_z"] = quaternions[:, 3]
    result["omega_body_x_radps"] = omegas[:, 0]
    result["omega_body_y_radps"] = omegas[:, 1]
    result["omega_body_z_radps"] = omegas[:, 2]
    result["yaw_eci_from_body_rad"] = euler_zyx_rad[:, 0]
    result["pitch_eci_from_body_rad"] = euler_zyx_rad[:, 1]
    result["roll_eci_from_body_rad"] = euler_zyx_rad[:, 2]
    result["yaw_eci_from_body_deg"] = np.rad2deg(euler_zyx_rad[:, 0])
    result["pitch_eci_from_body_deg"] = np.rad2deg(euler_zyx_rad[:, 1])
    result["roll_eci_from_body_deg"] = np.rad2deg(euler_zyx_rad[:, 2])
    result["Bx_body_T"] = b_body_t[:, 0]
    result["By_body_T"] = b_body_t[:, 1]
    result["Bz_body_T"] = b_body_t[:, 2]
    result["B_body_norm_T"] = np.linalg.norm(b_body_t, axis=1)
    result["det_R_eci_from_body"] = det_rotation
    result["RT_R_minus_I_fro"] = np.linalg.norm(rt_r_minus_i, axis=(1, 2))

    for row in range(3):
        for col in range(3):
            result[f"RT_R_minus_I_{row + 1}{col + 1}"] = rt_r_minus_i[:, row, col]

    return result
