"""Kalman-family attitude estimation interfaces."""

from simulation.estimation.AEKF import AEKF, AEKFConfig, DEFAULT_OMEGA_PROCESS_NOISE_STD_DEGPS
from simulation.estimation.quaternion_aekf import QuaternionAEKF, QuaternionAEKFConfig
from simulation.interfaces import KalmanFilter
from simulation.types import KalmanFilterEstimate, KalmanFilterInput

__all__ = [
    "AEKF",
    "AEKFConfig",
    "DEFAULT_OMEGA_PROCESS_NOISE_STD_DEGPS",
    "KalmanFilter",
    "KalmanFilterEstimate",
    "KalmanFilterInput",
    "QuaternionAEKF",
    "QuaternionAEKFConfig",
]
