"""Result assembly, persistence and sanity-check reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.types import SimulationResult

RESULT_COLUMNS = [
    "t_s",
    "t_utc",
    "x_eci_m",
    "y_eci_m",
    "z_eci_m",
    "vx_eci_mps",
    "vy_eci_mps",
    "vz_eci_mps",
    "r_norm_m",
    "v_norm_mps",
    "x_ecef_m",
    "y_ecef_m",
    "z_ecef_m",
    "lat_deg",
    "lon_deg",
    "alt_m",
    "lat_rad",
    "lon_rad",
    "alt_km",
    "B_N_nT",
    "B_E_nT",
    "B_D_nT",
    "Bx_ecef_T",
    "By_ecef_T",
    "Bz_ecef_T",
    "Bx_eci_T",
    "By_eci_T",
    "Bz_eci_T",
    "B_norm_T",
    "B_norm_nT",
    "q_eci_from_body_w",
    "q_eci_from_body_x",
    "q_eci_from_body_y",
    "q_eci_from_body_z",
    "q_eci_from_body_norm",
    "omega_body_x_radps",
    "omega_body_y_radps",
    "omega_body_z_radps",
    "yaw_eci_from_body_rad",
    "pitch_eci_from_body_rad",
    "roll_eci_from_body_rad",
    "yaw_eci_from_body_deg",
    "pitch_eci_from_body_deg",
    "roll_eci_from_body_deg",
    "Bx_body_T",
    "By_body_T",
    "Bz_body_T",
    "B_body_norm_T",
    "Bx_magnetometer_T",
    "By_magnetometer_T",
    "Bz_magnetometer_T",
    "B_magnetometer_norm_T",
    "det_R_eci_from_body",
    "RT_R_minus_I_fro",
    "RT_R_minus_I_11",
    "RT_R_minus_I_12",
    "RT_R_minus_I_13",
    "RT_R_minus_I_21",
    "RT_R_minus_I_22",
    "RT_R_minus_I_23",
    "RT_R_minus_I_31",
    "RT_R_minus_I_32",
    "RT_R_minus_I_33",
    "q_kalman_w",
    "q_kalman_x",
    "q_kalman_y",
    "q_kalman_z",
    "q_kalman_norm",
    "q_kalman_error_angle_rad",
    "q_kalman_error_angle_deg",
    "P_kalman_ww",
    "P_kalman_xx",
    "P_kalman_yy",
    "P_kalman_zz",
    "sigma_kalman_w",
    "sigma_kalman_x",
    "sigma_kalman_y",
    "sigma_kalman_z",
    "innovation_kalman_x_T",
    "innovation_kalman_y_T",
    "innovation_kalman_z_T",
    "innovation_kalman_norm_T",
]

KALMAN_RESULT_COLUMNS = RESULT_COLUMNS[RESULT_COLUMNS.index("q_kalman_w") :]


def build_results_dataframe(result: SimulationResult) -> pd.DataFrame:
    """Combine simulation states into a tabular result for export and plotting."""

    orbit = result.orbit
    frame = result.frame
    magnetic_field = result.magnetic_field
    attitude = result.attitude
    b_body_t = result.b_body_t
    b_magnetometer_t = result.b_magnetometer_t

    df = pd.DataFrame(
        {
            "t_s": orbit.t_s,
            "t_utc": orbit.t_utc.isot,
            "x_eci_m": orbit.r_eci_m[:, 0],
            "y_eci_m": orbit.r_eci_m[:, 1],
            "z_eci_m": orbit.r_eci_m[:, 2],
            "vx_eci_mps": orbit.v_eci_mps[:, 0],
            "vy_eci_mps": orbit.v_eci_mps[:, 1],
            "vz_eci_mps": orbit.v_eci_mps[:, 2],
            "x_ecef_m": frame.r_ecef_m[:, 0],
            "y_ecef_m": frame.r_ecef_m[:, 1],
            "z_ecef_m": frame.r_ecef_m[:, 2],
            "lat_deg": frame.lat_deg,
            "lon_deg": frame.lon_deg,
            "alt_m": frame.alt_m,
            "B_N_nT": magnetic_field.b_ned_nt[:, 0],
            "B_E_nT": magnetic_field.b_ned_nt[:, 1],
            "B_D_nT": magnetic_field.b_ned_nt[:, 2],
            "Bx_ecef_T": magnetic_field.b_ecef_t[:, 0],
            "By_ecef_T": magnetic_field.b_ecef_t[:, 1],
            "Bz_ecef_T": magnetic_field.b_ecef_t[:, 2],
            "Bx_eci_T": magnetic_field.b_eci_t[:, 0],
            "By_eci_T": magnetic_field.b_eci_t[:, 1],
            "Bz_eci_T": magnetic_field.b_eci_t[:, 2],
            "q_eci_from_body_w": attitude.q_eci_from_body[:, 0],
            "q_eci_from_body_x": attitude.q_eci_from_body[:, 1],
            "q_eci_from_body_y": attitude.q_eci_from_body[:, 2],
            "q_eci_from_body_z": attitude.q_eci_from_body[:, 3],
            "omega_body_x_radps": attitude.omega_body_radps[:, 0],
            "omega_body_y_radps": attitude.omega_body_radps[:, 1],
            "omega_body_z_radps": attitude.omega_body_radps[:, 2],
            "yaw_eci_from_body_rad": attitude.euler_zyx_rad[:, 0],
            "pitch_eci_from_body_rad": attitude.euler_zyx_rad[:, 1],
            "roll_eci_from_body_rad": attitude.euler_zyx_rad[:, 2],
            "Bx_body_T": b_body_t[:, 0],
            "By_body_T": b_body_t[:, 1],
            "Bz_body_T": b_body_t[:, 2],
            "Bx_magnetometer_T": b_magnetometer_t[:, 0],
            "By_magnetometer_T": b_magnetometer_t[:, 1],
            "Bz_magnetometer_T": b_magnetometer_t[:, 2],
            "det_R_eci_from_body": attitude.det_rotation,
        }
    )

    df["r_norm_m"] = np.linalg.norm(orbit.r_eci_m, axis=1)
    df["v_norm_mps"] = np.linalg.norm(orbit.v_eci_mps, axis=1)
    df["lat_rad"] = np.deg2rad(frame.lat_deg)
    df["lon_rad"] = np.deg2rad(frame.lon_deg)
    df["alt_km"] = frame.alt_m / 1000.0
    df["B_norm_T"] = np.linalg.norm(magnetic_field.b_eci_t, axis=1)
    df["B_norm_nT"] = df["B_norm_T"] * 1e9
    df["yaw_eci_from_body_deg"] = np.rad2deg(attitude.euler_zyx_rad[:, 0])
    df["pitch_eci_from_body_deg"] = np.rad2deg(attitude.euler_zyx_rad[:, 1])
    df["roll_eci_from_body_deg"] = np.rad2deg(attitude.euler_zyx_rad[:, 2])
    df["q_eci_from_body_norm"] = np.linalg.norm(attitude.q_eci_from_body, axis=1)
    df["B_body_norm_T"] = np.linalg.norm(b_body_t, axis=1)
    df["B_magnetometer_norm_T"] = np.linalg.norm(b_magnetometer_t, axis=1)
    df["RT_R_minus_I_fro"] = np.linalg.norm(attitude.rt_r_minus_i, axis=(1, 2))

    for row in range(3):
        for col in range(3):
            df[f"RT_R_minus_I_{row + 1}{col + 1}"] = attitude.rt_r_minus_i[:, row, col]

    _add_kalman_columns(df, result)

    return df[RESULT_COLUMNS]


def _add_kalman_columns(df: pd.DataFrame, result: SimulationResult) -> None:
    """Add Kalman estimate columns, using NaN placeholders when no filter ran."""

    for column in KALMAN_RESULT_COLUMNS:
        df[column] = np.nan

    estimate = result.kalman_estimate

    if estimate is None:
        return

    sample_count = len(df)
    states = np.asarray(estimate.state, dtype=np.float64)
    covariance = np.asarray(estimate.covariance, dtype=np.float64)

    if states.shape != (sample_count, 4):
        raise ValueError("Kalman estimate state must have shape (N, 4).")
    if covariance.shape != (sample_count, 4, 4):
        raise ValueError("Kalman estimate covariance must have shape (N, 4, 4).")

    df["q_kalman_w"] = states[:, 0]
    df["q_kalman_x"] = states[:, 1]
    df["q_kalman_y"] = states[:, 2]
    df["q_kalman_z"] = states[:, 3]
    df["q_kalman_norm"] = np.linalg.norm(states, axis=1)

    true_quaternions = result.attitude.q_eci_from_body
    denominator = np.linalg.norm(states, axis=1) * np.linalg.norm(true_quaternions, axis=1)
    cosine_half_angle = np.full(sample_count, np.nan, dtype=np.float64)
    np.divide(
        np.abs(np.einsum("ij,ij->i", states, true_quaternions)),
        denominator,
        out=cosine_half_angle,
        where=denominator > 0.0,
    )
    cosine_half_angle = np.clip(cosine_half_angle, -1.0, 1.0)
    df["q_kalman_error_angle_rad"] = 2.0 * np.arccos(cosine_half_angle)
    df["q_kalman_error_angle_deg"] = np.rad2deg(df["q_kalman_error_angle_rad"])

    covariance_diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    df["P_kalman_ww"] = covariance_diagonal[:, 0]
    df["P_kalman_xx"] = covariance_diagonal[:, 1]
    df["P_kalman_yy"] = covariance_diagonal[:, 2]
    df["P_kalman_zz"] = covariance_diagonal[:, 3]
    sigma = np.sqrt(np.maximum(covariance_diagonal, 0.0))
    df["sigma_kalman_w"] = sigma[:, 0]
    df["sigma_kalman_x"] = sigma[:, 1]
    df["sigma_kalman_y"] = sigma[:, 2]
    df["sigma_kalman_z"] = sigma[:, 3]

    if estimate.innovation is not None:
        innovation = np.asarray(estimate.innovation, dtype=np.float64)

        if innovation.shape != (sample_count, 3):
            raise ValueError("Kalman innovation must have shape (N, 3).")

        df["innovation_kalman_x_T"] = innovation[:, 0]
        df["innovation_kalman_y_T"] = innovation[:, 1]
        df["innovation_kalman_z_T"] = innovation[:, 2]
        df["innovation_kalman_norm_T"] = np.linalg.norm(innovation, axis=1)


def save_results(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save complete simulation results to CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "orbit_timeseries.csv"
    df.to_csv(output_path, index=False)

    return output_path


def print_sanity_check(df: pd.DataFrame) -> None:
    """Print basic orbit and attitude sanity-check values."""

    print()
    print("Sanity check:")
    print(f"Initial radius: {df['r_norm_m'].iloc[0] / 1000.0:.3f} km")
    print(f"Mean radius:    {df['r_norm_m'].mean() / 1000.0:.3f} km")
    print(f"Min radius:     {df['r_norm_m'].min() / 1000.0:.3f} km")
    print(f"Max radius:     {df['r_norm_m'].max() / 1000.0:.3f} km")
    print(f"Mean velocity:  {df['v_norm_mps'].mean() / 1000.0:.3f} km/s")

    orthogonality_columns = [
        f"RT_R_minus_I_{row}{col}" for row in range(1, 4) for col in range(1, 4)
    ]
    initial_orthogonality_error = df[orthogonality_columns].iloc[0].to_numpy().reshape(3, 3)
    final_orthogonality_error = df[orthogonality_columns].iloc[-1].to_numpy().reshape(3, 3)

    print()
    print("Attitude sanity check:")
    print("Initial R^T R - I:")
    for row in initial_orthogonality_error:
        print("  [" + " ".join(f"{value: .3e}" for value in row) + "]")
    print("Final R^T R - I:")
    for row in final_orthogonality_error:
        print("  [" + " ".join(f"{value: .3e}" for value in row) + "]")
    print(f"Max ||R^T R - I||_F: {df['RT_R_minus_I_fro'].max():.3e}")
    print(f"det(R) initial:      {df['det_R_eci_from_body'].iloc[0]:.12f}")
    print(f"det(R) final:        {df['det_R_eci_from_body'].iloc[-1]:.12f}")
    print(
        "det(R) min/max:      "
        f"{df['det_R_eci_from_body'].min():.12f} / {df['det_R_eci_from_body'].max():.12f}"
    )
