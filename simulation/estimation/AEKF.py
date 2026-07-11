"""Additive extended Kalman filter for quaternion-only attitude estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np

from simulation.attitude.rotations import quaternion_multiply, quaternion_to_rotation_matrix
from simulation.helpers import normalize_quaternion
from simulation.types import ArrayFloat64, KalmanFilterEstimate, KalmanFilterInput


def _default_quaternion() -> ArrayFloat64:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _default_initial_covariance() -> ArrayFloat64:
    return np.eye(4, dtype=np.float64) * 1e-3


def _default_process_noise() -> ArrayFloat64:
    return np.eye(4, dtype=np.float64) * 1e-9


def _default_measurement_noise() -> ArrayFloat64:
    return np.eye(3, dtype=np.float64) * (1e-6**2)


@dataclass(frozen=True, slots=True)
class AEKFConfig:
    """Configuration for a quaternion-only additive EKF."""

    initial_quaternion_eci_from_body: ArrayFloat64 = field(default_factory=_default_quaternion)
    initial_covariance: ArrayFloat64 = field(default_factory=_default_initial_covariance)
    process_noise: ArrayFloat64 = field(default_factory=_default_process_noise)
    measurement_noise: ArrayFloat64 = field(default_factory=_default_measurement_noise)
    jacobian_step: float = 1e-6


class AEKF:
    """
    Quaternion-state additive extended Kalman filter.

    The state is scalar-first ``q_eci_from_body``. The measurement model predicts
    the body-frame magnetometer sample from the inertial magnetic-field vector:
    ``B_body = R_eci_from_body(q).T @ B_eci``.
    """

    def __init__(self, config: AEKFConfig | None = None) -> None:
        self.config = config or AEKFConfig()
        self.initial_quaternion_eci_from_body = normalize_quaternion(
            np.asarray(self.config.initial_quaternion_eci_from_body, dtype=np.float64)
        )
        self.initial_covariance = _as_matrix(
            self.config.initial_covariance, (4, 4), "initial_covariance"
        )
        self.process_noise = _as_matrix(self.config.process_noise, (4, 4), "process_noise")
        self.measurement_noise = _as_matrix(
            self.config.measurement_noise, (3, 3), "measurement_noise"
        )
        self.jacobian_step = float(self.config.jacobian_step)

        if self.jacobian_step <= 0.0:
            raise ValueError("jacobian_step must be positive.")

        _validate_symmetric(self.initial_covariance, "initial_covariance")
        _validate_symmetric(self.process_noise, "process_noise")
        _validate_symmetric(self.measurement_noise, "measurement_noise")
        _validate_positive_semidefinite(self.initial_covariance, "initial_covariance")
        _validate_positive_semidefinite(self.process_noise, "process_noise")
        _validate_positive_definite(self.measurement_noise, "measurement_noise")

    def estimate(self, inputs: KalmanFilterInput) -> KalmanFilterEstimate:
        """Run prediction and update over a complete time series."""

        times_s, measurements_body_t, reference_vectors_eci_t, angular_rates_body_radps = (
            _validate_inputs(inputs)
        )
        sample_count = len(times_s)

        states = np.empty((sample_count, 4), dtype=np.float64)
        covariances = np.empty((sample_count, 4, 4), dtype=np.float64)
        innovations = np.empty((sample_count, 3), dtype=np.float64)
        innovation_covariances = np.empty((sample_count, 3, 3), dtype=np.float64)

        quaternion_plus = self.initial_quaternion_eci_from_body
        covariance_plus = self.initial_covariance

        for sample_index in range(sample_count):
            if sample_index == 0:
                quaternion_minus = quaternion_plus
                covariance_minus = covariance_plus
            else:
                dt_s = float(times_s[sample_index] - times_s[sample_index - 1])
                quaternion_minus, covariance_minus, _ = self.predict(
                    quaternion_plus,
                    covariance_plus,
                    angular_rates_body_radps[sample_index - 1],
                    dt_s,
                )

            quaternion_plus, covariance_plus, innovation, innovation_covariance = self.update(
                quaternion_minus,
                covariance_minus,
                measurements_body_t[sample_index],
                reference_vectors_eci_t[sample_index],
            )
            states[sample_index] = quaternion_plus
            covariances[sample_index] = covariance_plus
            innovations[sample_index] = innovation
            innovation_covariances[sample_index] = innovation_covariance

        return KalmanFilterEstimate(
            t_s=times_s,
            state=states,
            covariance=covariances,
            innovation=innovations,
            innovation_covariance=innovation_covariances,
        )

    def predict(
        self,
        quaternion_plus: ArrayFloat64,
        covariance_plus: ArrayFloat64,
        omega_body_radps: ArrayFloat64,
        dt_s: float,
    ) -> tuple[ArrayFloat64, ArrayFloat64, ArrayFloat64]:
        """Predict ``q_k^-`` and ``P_k^-`` from the previous posterior state."""

        quaternion_minus = self.predict_quaternion(quaternion_plus, omega_body_radps, dt_s)
        transition_jacobian = self.state_transition_jacobian(
            quaternion_plus, omega_body_radps, dt_s
        )
        covariance = _as_matrix(covariance_plus, (4, 4), "covariance_plus")
        covariance_minus = (
            transition_jacobian @ covariance @ transition_jacobian.T + self.process_noise
        )

        return quaternion_minus, _symmetrize(covariance_minus), transition_jacobian

    def update(
        self,
        quaternion_minus: ArrayFloat64,
        covariance_minus: ArrayFloat64,
        measurement_body_t: ArrayFloat64,
        reference_vector_eci_t: ArrayFloat64,
    ) -> tuple[ArrayFloat64, ArrayFloat64, ArrayFloat64, ArrayFloat64]:
        """Apply the magnetometer measurement update."""

        quaternion = normalize_quaternion(np.asarray(quaternion_minus, dtype=np.float64))
        covariance = _as_matrix(covariance_minus, (4, 4), "covariance_minus")
        measurement = _as_vector(measurement_body_t, 3, "measurement_body_t")
        reference = _as_vector(reference_vector_eci_t, 3, "reference_vector_eci_t")

        predicted_measurement = self.predict_measurement(quaternion, reference)
        innovation = measurement - predicted_measurement
        measurement_jacobian = self.measurement_jacobian(quaternion, reference)
        innovation_covariance = (
            measurement_jacobian @ covariance @ measurement_jacobian.T + self.measurement_noise
        )
        gain = np.linalg.solve(innovation_covariance, measurement_jacobian @ covariance).T

        quaternion_plus = normalize_quaternion(quaternion + gain @ innovation)
        identity = np.eye(4, dtype=np.float64)
        joseph_factor = identity - gain @ measurement_jacobian
        covariance_plus = (
            joseph_factor @ covariance @ joseph_factor.T + gain @ self.measurement_noise @ gain.T
        )

        return (
            quaternion_plus,
            _symmetrize(covariance_plus),
            innovation,
            _symmetrize(innovation_covariance),
        )

    def predict_quaternion(
        self, quaternion: ArrayFloat64, omega_body_radps: ArrayFloat64, dt_s: float
    ) -> ArrayFloat64:
        """Compute ``f(q, omega, dt)`` for the quaternion-only process model."""

        if dt_s < 0.0:
            raise ValueError("dt_s must be nonnegative.")

        quaternion = normalize_quaternion(np.asarray(quaternion, dtype=np.float64))
        omega = _as_vector(omega_body_radps, 3, "omega_body_radps")
        omega_norm = float(np.linalg.norm(omega))

        if omega_norm == 0.0 or dt_s == 0.0:
            delta_quaternion = _default_quaternion()
        else:
            half_angle = 0.5 * omega_norm * dt_s
            vector_scale = np.sin(half_angle) / omega_norm
            delta_quaternion = np.array(
                [
                    np.cos(half_angle),
                    omega[0] * vector_scale,
                    omega[1] * vector_scale,
                    omega[2] * vector_scale,
                ],
                dtype=np.float64,
            )

        return normalize_quaternion(quaternion_multiply(quaternion, delta_quaternion))

    def state_transition_jacobian(
        self, quaternion: ArrayFloat64, omega_body_radps: ArrayFloat64, dt_s: float
    ) -> ArrayFloat64:
        """Compute ``F_k = df/dq`` with central finite differences."""

        omega = _as_vector(omega_body_radps, 3, "omega_body_radps")
        return _finite_difference_jacobian(
            lambda state: self.predict_quaternion(state, omega, dt_s),
            normalize_quaternion(np.asarray(quaternion, dtype=np.float64)),
            4,
            self.jacobian_step,
        )

    def predict_measurement(
        self, quaternion: ArrayFloat64, reference_vector_eci_t: ArrayFloat64
    ) -> ArrayFloat64:
        """Compute ``h(q)`` for a body-frame magnetic-field measurement."""

        rotation_eci_from_body = quaternion_to_rotation_matrix(quaternion)
        reference = _as_vector(reference_vector_eci_t, 3, "reference_vector_eci_t")

        return rotation_eci_from_body.T @ reference

    def measurement_jacobian(
        self, quaternion: ArrayFloat64, reference_vector_eci_t: ArrayFloat64
    ) -> ArrayFloat64:
        """Compute ``H_k = dh/dq`` with central finite differences."""

        reference = _as_vector(reference_vector_eci_t, 3, "reference_vector_eci_t")
        return _finite_difference_jacobian(
            lambda state: self.predict_measurement(state, reference),
            normalize_quaternion(np.asarray(quaternion, dtype=np.float64)),
            3,
            self.jacobian_step,
        )


def _validate_inputs(
    inputs: KalmanFilterInput,
) -> tuple[ArrayFloat64, ArrayFloat64, ArrayFloat64, ArrayFloat64]:
    times_s = np.asarray(inputs.t_s, dtype=np.float64)
    measurements_body_t = np.asarray(inputs.measurements_body_t, dtype=np.float64)
    reference_vectors_eci_t = np.asarray(inputs.reference_vectors_eci_t, dtype=np.float64)

    if times_s.ndim != 1:
        raise ValueError("t_s must be a one-dimensional array.")
    if len(times_s) == 0:
        raise ValueError("t_s must contain at least one sample.")
    if np.any(np.diff(times_s) < 0.0):
        raise ValueError("t_s must be monotonically nondecreasing.")
    if measurements_body_t.shape != (len(times_s), 3):
        raise ValueError("measurements_body_t must have shape (N, 3).")
    if reference_vectors_eci_t.shape != (len(times_s), 3):
        raise ValueError("reference_vectors_eci_t must have shape (N, 3).")

    if inputs.angular_rate_body_radps is None:
        angular_rates_body_radps = np.zeros((len(times_s), 3), dtype=np.float64)
    else:
        angular_rates_body_radps = np.asarray(inputs.angular_rate_body_radps, dtype=np.float64)
        if angular_rates_body_radps.shape != (len(times_s), 3):
            raise ValueError("angular_rate_body_radps must have shape (N, 3).")

    return times_s, measurements_body_t, reference_vectors_eci_t, angular_rates_body_radps


def _finite_difference_jacobian(
    function: Callable[[ArrayFloat64], ArrayFloat64],
    point: ArrayFloat64,
    output_size: int,
    step: float,
) -> ArrayFloat64:
    jacobian = np.empty((output_size, len(point)), dtype=np.float64)

    for column in range(len(point)):
        perturbation = np.zeros(len(point), dtype=np.float64)
        perturbation[column] = step
        upper = function(point + perturbation)
        lower = function(point - perturbation)
        jacobian[:, column] = (upper - lower) / (2.0 * step)

    return jacobian


def _as_vector(values: ArrayFloat64, length: int, name: str) -> ArrayFloat64:
    vector = np.asarray(values, dtype=np.float64)

    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},).")

    return vector


def _as_matrix(values: ArrayFloat64, shape: tuple[int, int], name: str) -> ArrayFloat64:
    matrix = np.asarray(values, dtype=np.float64)

    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")

    return matrix


def _validate_symmetric(matrix: ArrayFloat64, name: str) -> None:
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")


def _validate_positive_semidefinite(matrix: ArrayFloat64, name: str) -> None:
    if np.any(np.linalg.eigvalsh(matrix) < -1e-15):
        raise ValueError(f"{name} must be positive semidefinite.")


def _validate_positive_definite(matrix: ArrayFloat64, name: str) -> None:
    if np.any(np.linalg.eigvalsh(matrix) <= 0.0):
        raise ValueError(f"{name} must be positive definite.")


def _symmetrize(matrix: ArrayFloat64) -> ArrayFloat64:
    return 0.5 * (matrix + matrix.T)
