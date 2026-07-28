"""Additive extended Kalman filter for attitude, body rate and magnetometer bias."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from simulation.attitude.dynamics import rigid_body_derivative
from simulation.attitude.rotations import quaternion_multiply, quaternion_to_rotation_matrix
from simulation.helpers import normalize_quaternion
from simulation.types import ArrayFloat64, KalmanFilterEstimate, KalmanFilterInput

QUATERNION_SLICE = slice(0, 4)
OMEGA_SLICE = slice(4, 7)
MAGNETOMETER_BIAS_SLICE = slice(7, 10)
STATE_SIZE = 10
MEASUREMENT_SIZE = 3


def _default_quaternion() -> ArrayFloat64:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _default_vector() -> ArrayFloat64:
    return np.zeros(3, dtype=np.float64)


def _default_initial_covariance() -> ArrayFloat64:
    diagonal = np.array(
        [
            1.0e-3,
            1.0e-3,
            1.0e-3,
            1.0e-3,
            *(np.deg2rad([0.05, 0.05, 0.05]) ** 2),
            *((0.5e-6) ** 2 * np.ones(3, dtype=np.float64)),
        ],
        dtype=np.float64,
    )

    return np.diag(diagonal)


def _default_process_noise() -> ArrayFloat64:
    diagonal = np.array(
        [
            1.0e-10,
            1.0e-10,
            1.0e-10,
            1.0e-10,
            *(np.deg2rad([2.0e-4, 2.0e-4, 2.0e-4]) ** 2),
            *((1.0e-10) ** 2 * np.ones(3, dtype=np.float64)),
        ],
        dtype=np.float64,
    )

    return np.diag(diagonal)


def _default_measurement_noise() -> ArrayFloat64:
    return np.eye(3, dtype=np.float64) * (1.0e-6**2)


def _default_inertia() -> ArrayFloat64:
    return np.eye(3, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AEKFConfig:
    """Configuration for a magnetometer-only 10-state additive EKF."""

    initial_quaternion_eci_from_body: ArrayFloat64 = field(default_factory=_default_quaternion)
    initial_omega_body_radps: ArrayFloat64 = field(default_factory=_default_vector)
    initial_magnetometer_bias_body_t: ArrayFloat64 = field(default_factory=_default_vector)
    inertia_kg_m2: ArrayFloat64 = field(default_factory=_default_inertia)
    torque_body_nm: ArrayFloat64 = field(default_factory=_default_vector)
    initial_covariance: ArrayFloat64 = field(default_factory=_default_initial_covariance)
    process_noise: ArrayFloat64 = field(default_factory=_default_process_noise)
    measurement_noise: ArrayFloat64 = field(default_factory=_default_measurement_noise)
    jacobian_step: float = 1.0e-6


class AEKF:
    """
    Additive EKF with state ``[q, omega_body, magnetometer_bias_body]``.

    ``q`` is scalar-first and maps body coordinates into ECI. The only
    measurement is the body-frame magnetic-field vector:
    ``B_body = R_eci_from_body(q).T @ B_eci + b_mag``.
    """

    def __init__(self, config: AEKFConfig | None = None) -> None:
        self.config = config or AEKFConfig()
        self.initial_state = _pack_state(
            normalize_quaternion(
                np.asarray(self.config.initial_quaternion_eci_from_body, dtype=np.float64)
            ),
            _as_vector(self.config.initial_omega_body_radps, 3, "initial_omega_body_radps"),
            _as_vector(
                self.config.initial_magnetometer_bias_body_t, 3, "initial_magnetometer_bias_body_t"
            ),
        )
        self.initial_covariance = _as_matrix(
            self.config.initial_covariance, (STATE_SIZE, STATE_SIZE), "initial_covariance"
        )
        self.process_noise = _as_matrix(
            self.config.process_noise, (STATE_SIZE, STATE_SIZE), "process_noise"
        )
        self.inertia_kg_m2 = _as_matrix(self.config.inertia_kg_m2, (3, 3), "inertia_kg_m2")
        self.torque_body_nm = _as_vector(self.config.torque_body_nm, 3, "torque_body_nm")
        self.measurement_noise = _as_matrix(
            self.config.measurement_noise, (MEASUREMENT_SIZE, MEASUREMENT_SIZE), "measurement_noise"
        )
        self.jacobian_step = float(self.config.jacobian_step)

        if self.jacobian_step <= 0.0:
            raise ValueError("jacobian_step must be positive.")

        _validate_covariance(self.initial_covariance, "initial_covariance", positive_definite=False)
        _validate_covariance(self.process_noise, "process_noise", positive_definite=False)
        _validate_covariance(self.measurement_noise, "measurement_noise", positive_definite=True)
        _validate_covariance(self.inertia_kg_m2, "inertia_kg_m2", positive_definite=True)

    def estimate(self, inputs: KalmanFilterInput) -> KalmanFilterEstimate:
        """Run prediction and magnetometer update over a complete time series."""

        times_s, measurements_body_t, reference_vectors_eci_t = _validate_inputs(inputs)
        sample_count = len(times_s)

        states = np.empty((sample_count, STATE_SIZE), dtype=np.float64)
        covariances = np.empty((sample_count, STATE_SIZE, STATE_SIZE), dtype=np.float64)
        innovations = np.empty((sample_count, MEASUREMENT_SIZE), dtype=np.float64)
        innovation_covariances = np.empty(
            (sample_count, MEASUREMENT_SIZE, MEASUREMENT_SIZE), dtype=np.float64
        )

        state_plus = self.initial_state
        covariance_plus = self.initial_covariance

        for sample_index in range(sample_count):
            if sample_index == 0:
                state_minus = state_plus
                covariance_minus = covariance_plus
            else:
                dt_s = float(times_s[sample_index] - times_s[sample_index - 1])
                state_minus, covariance_minus, _ = self.predict(state_plus, covariance_plus, dt_s)

            state_plus, covariance_plus, innovation, innovation_covariance = self.update(
                state_minus,
                covariance_minus,
                measurements_body_t[sample_index],
                reference_vectors_eci_t[sample_index],
            )
            states[sample_index] = state_plus
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
        self, state_plus: ArrayFloat64, covariance_plus: ArrayFloat64, dt_s: float
    ) -> tuple[ArrayFloat64, ArrayFloat64, ArrayFloat64]:
        """Predict ``x_k^-`` and ``P_k^-`` from the previous posterior state."""

        state = _as_vector(state_plus, STATE_SIZE, "state_plus")
        covariance = _as_matrix(covariance_plus, (STATE_SIZE, STATE_SIZE), "covariance_plus")
        state_minus = self.predict_state(state, dt_s)
        transition_jacobian = self.state_transition_jacobian(state, dt_s)
        covariance_minus = (
            transition_jacobian @ covariance @ transition_jacobian.T + self.process_noise
        )

        return state_minus, _symmetrize(covariance_minus), transition_jacobian

    def update(
        self,
        state_minus: ArrayFloat64,
        covariance_minus: ArrayFloat64,
        measurement_body_t: ArrayFloat64,
        reference_vector_eci_t: ArrayFloat64,
    ) -> tuple[ArrayFloat64, ArrayFloat64, ArrayFloat64, ArrayFloat64]:
        """Apply the magnetometer-only measurement update."""

        state = _normalize_state(_as_vector(state_minus, STATE_SIZE, "state_minus"))
        covariance = _as_matrix(covariance_minus, (STATE_SIZE, STATE_SIZE), "covariance_minus")
        measurement = _as_vector(measurement_body_t, MEASUREMENT_SIZE, "measurement_body_t")
        reference = _as_vector(reference_vector_eci_t, 3, "reference_vector_eci_t")

        predicted_measurement = self.predict_measurement_from_state(state, reference)
        innovation = measurement - predicted_measurement
        measurement_jacobian = self.measurement_jacobian(state, reference)
        innovation_covariance = (
            measurement_jacobian @ covariance @ measurement_jacobian.T + self.measurement_noise
        )
        gain = np.linalg.solve(innovation_covariance, measurement_jacobian @ covariance).T

        state_plus = _normalize_state(state + gain @ innovation)
        identity = np.eye(STATE_SIZE, dtype=np.float64)
        joseph_factor = identity - gain @ measurement_jacobian
        covariance_plus = (
            joseph_factor @ covariance @ joseph_factor.T + gain @ self.measurement_noise @ gain.T
        )

        return (
            state_plus,
            _symmetrize(covariance_plus),
            innovation,
            _symmetrize(innovation_covariance),
        )

    def predict_state(self, state: ArrayFloat64, dt_s: float) -> ArrayFloat64:
        """Propagate quaternion while treating omega and mag bias as random walks."""

        if dt_s < 0.0:
            raise ValueError("dt_s must be nonnegative.")

        quaternion, omega_body_radps, magnetometer_bias_body_t = _unpack_state(state)
        quaternion_minus, omega_minus = self.propagate_attitude_state(
            quaternion, omega_body_radps, dt_s
        )

        return _pack_state(quaternion_minus, omega_minus, magnetometer_bias_body_t)

    def propagate_attitude_state(
        self, quaternion: ArrayFloat64, omega_body_radps: ArrayFloat64, dt_s: float
    ) -> tuple[ArrayFloat64, ArrayFloat64]:
        """Propagate quaternion and angular velocity with rigid-body dynamics."""

        if dt_s < 0.0:
            raise ValueError("dt_s must be nonnegative.")

        state = np.concatenate(
            (
                normalize_quaternion(np.asarray(quaternion, dtype=np.float64)),
                _as_vector(omega_body_radps, 3, "omega_body_radps"),
            )
        )

        if dt_s == 0.0:
            return state[:4], state[4:]

        def derivative(attitude_state: ArrayFloat64) -> ArrayFloat64:
            q = normalize_quaternion(attitude_state[:4])
            omega = attitude_state[4:]
            q_dot, omega_dot = rigid_body_derivative(
                q, omega, self.inertia_kg_m2, self.torque_body_nm
            )

            return np.concatenate((q_dot, omega_dot))

        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt_s * k1)
        k3 = derivative(state + 0.5 * dt_s * k2)
        k4 = derivative(state + dt_s * k3)
        propagated = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return normalize_quaternion(propagated[:4]), propagated[4:]

    def predict_quaternion(
        self, quaternion: ArrayFloat64, omega_body_radps: ArrayFloat64, dt_s: float
    ) -> ArrayFloat64:
        """Compute ``f(q, omega, dt)`` for the attitude process model."""

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

    def state_transition_jacobian(self, state: ArrayFloat64, dt_s: float) -> ArrayFloat64:
        """Compute ``F_k = df/dx`` with central finite differences."""

        return _finite_difference_jacobian(
            lambda perturbed_state: self.predict_state(perturbed_state, dt_s),
            _normalize_state(_as_vector(state, STATE_SIZE, "state")),
            STATE_SIZE,
            self.jacobian_step,
        )

    def predict_measurement(
        self, quaternion: ArrayFloat64, reference_vector_eci_t: ArrayFloat64
    ) -> ArrayFloat64:
        """Compute the bias-free magnetometer measurement ``h_B(q)``."""

        rotation_eci_from_body = quaternion_to_rotation_matrix(quaternion)
        reference = _as_vector(reference_vector_eci_t, 3, "reference_vector_eci_t")

        return rotation_eci_from_body.T @ reference

    def predict_measurement_from_state(
        self, state: ArrayFloat64, reference_vector_eci_t: ArrayFloat64
    ) -> ArrayFloat64:
        """Compute ``h(x)`` for a magnetometer sample."""

        quaternion, _omega_body_radps, magnetometer_bias_body_t = _unpack_state(state)

        return (
            self.predict_measurement(quaternion, reference_vector_eci_t) + magnetometer_bias_body_t
        )

    def measurement_jacobian(
        self, state: ArrayFloat64, reference_vector_eci_t: ArrayFloat64
    ) -> ArrayFloat64:
        """Compute ``H_k = dh/dx`` with central finite differences."""

        reference = _as_vector(reference_vector_eci_t, 3, "reference_vector_eci_t")
        return _finite_difference_jacobian(
            lambda perturbed_state: self.predict_measurement_from_state(perturbed_state, reference),
            _normalize_state(_as_vector(state, STATE_SIZE, "state")),
            MEASUREMENT_SIZE,
            self.jacobian_step,
        )


def _validate_inputs(inputs: KalmanFilterInput) -> tuple[ArrayFloat64, ArrayFloat64, ArrayFloat64]:
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

    return times_s, measurements_body_t, reference_vectors_eci_t


def _pack_state(
    quaternion_eci_from_body: ArrayFloat64,
    omega_body_radps: ArrayFloat64,
    magnetometer_bias_body_t: ArrayFloat64,
) -> ArrayFloat64:
    return np.concatenate(
        (
            normalize_quaternion(np.asarray(quaternion_eci_from_body, dtype=np.float64)),
            _as_vector(omega_body_radps, 3, "omega_body_radps"),
            _as_vector(magnetometer_bias_body_t, 3, "magnetometer_bias_body_t"),
        )
    )


def _unpack_state(state: ArrayFloat64) -> tuple[ArrayFloat64, ArrayFloat64, ArrayFloat64]:
    state = _as_vector(state, STATE_SIZE, "state")

    return (
        normalize_quaternion(state[QUATERNION_SLICE]),
        state[OMEGA_SLICE],
        state[MAGNETOMETER_BIAS_SLICE],
    )


def _normalize_state(state: ArrayFloat64) -> ArrayFloat64:
    normalized = np.asarray(state, dtype=np.float64).copy()
    normalized[QUATERNION_SLICE] = normalize_quaternion(normalized[QUATERNION_SLICE])

    return normalized


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


def _validate_covariance(matrix: ArrayFloat64, name: str, *, positive_definite: bool) -> None:
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(matrix)

    if positive_definite:
        if np.any(eigenvalues <= 0.0):
            raise ValueError(f"{name} must be positive definite.")
    elif np.any(eigenvalues < -1.0e-15):
        raise ValueError(f"{name} must be positive semidefinite.")


def _symmetrize(matrix: ArrayFloat64) -> ArrayFloat64:
    return 0.5 * (matrix + matrix.T)
