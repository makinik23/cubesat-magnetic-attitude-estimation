"""AEKF validation runs, metrics and report generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from poliastro.bodies import Earth
from scipy.stats import chi2

from simulation.attitude import quaternion_multiply
from simulation.config import (
    create_default_attitude_config,
    create_default_magnetometer_config,
    create_default_orbit,
    create_default_simulation_config,
)
from simulation.estimation import AEKF, AEKFConfig
from simulation.helpers import normalize_quaternion
from simulation.io import build_results_dataframe, save_results
from simulation.pipeline import (
    AEKF_OUTPUT_DIR,
    KALMAN_CSV_FILENAME,
    SimulationRunner,
    save_general_plot_outputs,
    save_kalman_plot_outputs,
    save_kalman_results,
)
from simulation.sensors import MagnetometerModel
from simulation.types import (
    ArrayFloat64,
    AttitudeConfig,
    ClassicalOrbitalElements,
    KalmanFilterEstimate,
    KalmanFilterInput,
    MagnetometerConfig,
    SimulationConfig,
    SimulationResult,
)

DEFAULT_VALIDATION_OUTPUT_DIR = Path("outputs") / "validation"
ATTITUDE_SETTLING_THRESHOLDS_DEG = (2.0, 1.5, 1.0)
BIAS_SETTLING_THRESHOLDS_UT = (0.3, 0.2, 0.15, 0.1)
NIS_DIMENSION = 3


@dataclass(frozen=True, slots=True)
class AEKFValidationConfig:
    """Configuration for the default AEKF validation workflow."""

    monte_carlo_runs: int = 16
    random_seed: int = 20260820
    initial_attitude_error_max_deg: float = 5.0
    initial_omega_error_std_degps: float = 0.05
    initial_bias_estimate_std_uT: float = 0.5
    long_run_orbits: float = 5.0


DEFAULT_VALIDATION_CONFIG = AEKFValidationConfig()


def run_aekf_validation(
    output_dir: Path = DEFAULT_VALIDATION_OUTPUT_DIR, config: AEKFValidationConfig | None = None
) -> dict[str, Path]:
    """Run Monte Carlo, long-run validation and write summary artifacts."""

    validation_config = config or AEKFValidationConfig()
    if validation_config.monte_carlo_runs <= 0:
        raise ValueError("monte_carlo_runs must be positive.")
    if validation_config.long_run_orbits <= 0.0:
        raise ValueError("long_run_orbits must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    elements = create_default_orbit()
    simulation_config = create_default_simulation_config()
    attitude_config = create_default_attitude_config()
    magnetometer_config = create_default_magnetometer_config()
    period_s = orbital_period_s(elements)

    base_result = _build_base_truth_result(
        elements, simulation_config, attitude_config, magnetometer_config
    )
    monte_carlo_df = run_monte_carlo(
        base_result, attitude_config, magnetometer_config, validation_config, period_s
    )
    monte_carlo_csv = output_dir / "monte_carlo_summary.csv"
    monte_carlo_df.to_csv(monte_carlo_csv, index=False)
    save_monte_carlo_plots(monte_carlo_df, output_dir)
    monte_carlo_summary = summarize_monte_carlo(monte_carlo_df, period_s)
    monte_carlo_summary_json = output_dir / "monte_carlo_summary.json"
    _write_json(monte_carlo_summary_json, monte_carlo_summary)

    long_run_dir = output_dir / "long_run"
    long_df = run_long_aekf(
        elements,
        simulation_config,
        attitude_config,
        magnetometer_config,
        validation_config.long_run_orbits,
        period_s,
        long_run_dir,
    )
    long_summary = summarize_aekf_dataframe(long_df, magnetometer_config.bias_sensor_t, period_s)
    long_summary_json = long_run_dir / "summary.json"
    _write_json(long_summary_json, long_summary)

    report_path = output_dir / "aekf_validation_report.md"
    write_validation_report(
        report_path,
        validation_config,
        magnetometer_config,
        period_s,
        monte_carlo_summary,
        long_summary,
    )

    return {
        "monte_carlo_csv": monte_carlo_csv,
        "monte_carlo_summary_json": monte_carlo_summary_json,
        "long_run_csv": long_run_dir / AEKF_OUTPUT_DIR / KALMAN_CSV_FILENAME,
        "long_run_summary_json": long_summary_json,
        "report": report_path,
    }


def orbital_period_s(elements: ClassicalOrbitalElements) -> float:
    """Return the Keplerian orbital period implied by the semi-major axis."""

    semi_major_axis_m = float(elements.semi_major_axis.to_value(u.m))
    mu_earth_m3_s2 = float(Earth.k.to(u.m**3 / u.s**2).value)

    return float(2.0 * np.pi * np.sqrt(semi_major_axis_m**3 / mu_earth_m3_s2))


def run_monte_carlo(
    base_result: SimulationResult,
    attitude_config: AttitudeConfig,
    magnetometer_config: MagnetometerConfig,
    validation_config: AEKFValidationConfig,
    period_s: float,
) -> pd.DataFrame:
    """Run AEKF Monte Carlo perturbations against one truth trajectory."""

    rng = np.random.default_rng(validation_config.random_seed)
    rows: list[dict[str, Any]] = []

    for run_index in range(validation_config.monte_carlo_runs):
        noise_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        initial_attitude_error_deg = float(
            rng.uniform(0.0, validation_config.initial_attitude_error_max_deg)
        )
        initial_quaternion = _perturb_quaternion(
            attitude_config.initial_quaternion_eci_from_body,
            rng,
            np.deg2rad(initial_attitude_error_deg),
        )
        omega_error_radps = np.deg2rad(
            rng.normal(0.0, validation_config.initial_omega_error_std_degps, size=3)
        )
        initial_omega = attitude_config.initial_omega_body_radps + omega_error_radps
        initial_bias = rng.normal(
            0.0, validation_config.initial_bias_estimate_std_uT * 1.0e-6, size=3
        )

        measurement = MagnetometerModel(
            bias_sensor_t=magnetometer_config.bias_sensor_t,
            noise_std_t=magnetometer_config.noise_std_t,
            seed=noise_seed,
            sensor_axes_from_body=magnetometer_config.sensor_axes_from_body,
            positions_body_m=magnetometer_config.positions_body_m,
        ).measure(base_result.b_body_t)
        estimate = _estimate_aekf(
            base_result,
            attitude_config,
            magnetometer_config,
            measurement,
            initial_quaternion,
            initial_omega,
            initial_bias,
        )
        run_result = replace(base_result, b_magnetometer_t=measurement, kalman_estimate=estimate)
        run_df = build_results_dataframe(run_result)
        summary = summarize_aekf_dataframe(run_df, magnetometer_config.bias_sensor_t, period_s)
        bias_error_uT = (initial_bias - magnetometer_config.bias_sensor_t) * 1.0e6

        summary.update(
            {
                "run_index": run_index,
                "noise_seed": noise_seed,
                "initial_attitude_error_deg": initial_attitude_error_deg,
                "initial_omega_error_norm_degps": float(
                    np.linalg.norm(np.rad2deg(omega_error_radps))
                ),
                "initial_bias_error_norm_uT": float(np.linalg.norm(bias_error_uT)),
            }
        )
        rows.append(summary)

    return pd.DataFrame(rows)


def run_long_aekf(
    elements: ClassicalOrbitalElements,
    simulation_config: SimulationConfig,
    attitude_config: AttitudeConfig,
    magnetometer_config: MagnetometerConfig,
    orbit_count: float,
    period_s: float,
    output_dir: Path,
) -> pd.DataFrame:
    """Run and save a longer AEKF validation trajectory."""

    long_config = replace(simulation_config, duration_s=orbit_count * period_s)
    runner = SimulationRunner(
        magnetometer_model=MagnetometerModel(
            bias_sensor_t=magnetometer_config.bias_sensor_t,
            noise_std_t=magnetometer_config.noise_std_t,
            seed=magnetometer_config.seed,
            sensor_axes_from_body=magnetometer_config.sensor_axes_from_body,
            positions_body_m=magnetometer_config.positions_body_m,
        ),
        kalman_filter=AEKF(
            AEKFConfig(
                initial_quaternion_eci_from_body=attitude_config.initial_quaternion_eci_from_body,
                initial_omega_body_radps=attitude_config.initial_omega_body_radps,
                inertia_kg_m2=attitude_config.inertia_kg_m2,
                torque_body_nm=attitude_config.torque_body_nm,
                sensor_axes_from_body=magnetometer_config.sensor_axes_from_body,
                measurement_noise=_measurement_noise_from_sensor_config(magnetometer_config),
            )
        ),
    )
    result = runner.run(elements, long_config, attitude_config)
    df = build_results_dataframe(result)

    save_results(df, output_dir)
    kalman_output_dir = output_dir / AEKF_OUTPUT_DIR
    save_kalman_results(df, kalman_output_dir)
    save_general_plot_outputs(df, output_dir)
    save_kalman_plot_outputs(df, kalman_output_dir)

    return df


def summarize_aekf_dataframe(
    df: pd.DataFrame, true_bias_sensor_t: ArrayFloat64, period_s: float
) -> dict[str, Any]:
    """Compute scalar AEKF performance metrics from an exported result table."""

    times_s = df["t_s"].to_numpy(dtype=np.float64)
    attitude_error_deg = df["q_kalman_error_angle_deg"].to_numpy(dtype=np.float64)
    innovation_norm_uT = df["innovation_kalman_norm_T"].to_numpy(dtype=np.float64) * 1.0e6
    bias_estimate_uT = df[
        ["mag_bias_kalman_x_uT", "mag_bias_kalman_y_uT", "mag_bias_kalman_z_uT"]
    ].to_numpy(dtype=np.float64)
    true_bias_uT = np.asarray(true_bias_sensor_t, dtype=np.float64) * 1.0e6
    bias_error_norm_uT = np.linalg.norm(bias_estimate_uT - true_bias_uT, axis=1)
    nis = df["nis_kalman"].to_numpy(dtype=np.float64)
    nis_lower_95 = float(chi2.ppf(0.025, df=NIS_DIMENSION))
    nis_upper_95 = float(chi2.ppf(0.975, df=NIS_DIMENSION))

    summary: dict[str, Any] = {
        "sample_count": int(len(df)),
        "duration_s": float(times_s[-1] - times_s[0]),
        "duration_orbits": float((times_s[-1] - times_s[0]) / period_s),
        "attitude_final_deg": float(attitude_error_deg[-1]),
        "attitude_mean_deg": float(np.mean(attitude_error_deg)),
        "attitude_median_deg": float(np.median(attitude_error_deg)),
        "attitude_rms_deg": float(np.sqrt(np.mean(attitude_error_deg**2))),
        "attitude_max_deg": float(np.max(attitude_error_deg)),
        "bias_final_error_norm_uT": float(bias_error_norm_uT[-1]),
        "bias_best_error_norm_uT": float(np.min(bias_error_norm_uT)),
        "bias_best_time_s": float(times_s[int(np.argmin(bias_error_norm_uT))]),
        "innovation_final_norm_uT": float(innovation_norm_uT[-1]),
        "innovation_mean_norm_uT": float(np.mean(innovation_norm_uT)),
        "innovation_max_norm_uT": float(np.max(innovation_norm_uT)),
        "nis_mean": float(np.mean(nis)),
        "nis_median": float(np.median(nis)),
        "nis_p95": float(np.percentile(nis, 95.0)),
        "nis_central_95_fraction": float(np.mean((nis >= nis_lower_95) & (nis <= nis_upper_95))),
        "nis_below_upper_95_fraction": float(np.mean(nis <= nis_upper_95)),
    }

    omega_error_column = "omega_kalman_error_norm_degps"
    if omega_error_column in df:
        omega_error_degps = df[omega_error_column].to_numpy(dtype=np.float64)
        summary.update(
            {
                "omega_error_final_degps": float(omega_error_degps[-1]),
                "omega_error_mean_degps": float(np.mean(omega_error_degps)),
                "omega_error_max_degps": float(np.max(omega_error_degps)),
            }
        )

    for threshold in ATTITUDE_SETTLING_THRESHOLDS_DEG:
        key = f"attitude_settled_below_{_threshold_label(threshold)}_deg_s"
        summary[key] = _settling_time_s(times_s, attitude_error_deg, threshold)

    for threshold in BIAS_SETTLING_THRESHOLDS_UT:
        key = f"bias_settled_below_{_threshold_label(threshold)}_uT_s"
        summary[key] = _settling_time_s(times_s, bias_error_norm_uT, threshold)

    return summary


def summarize_monte_carlo(
    summary_df: pd.DataFrame, period_s: float | None = None
) -> dict[str, Any]:
    """Aggregate per-run Monte Carlo metrics into one compact summary."""

    summary = {
        "runs": int(len(summary_df)),
        "attitude_final_deg_mean": _column_mean(summary_df, "attitude_final_deg"),
        "attitude_final_deg_p50": _column_quantile(summary_df, "attitude_final_deg", 0.5),
        "attitude_final_deg_p95": _column_quantile(summary_df, "attitude_final_deg", 0.95),
        "attitude_rms_deg_mean": _column_mean(summary_df, "attitude_rms_deg"),
        "bias_final_error_norm_uT_mean": _column_mean(summary_df, "bias_final_error_norm_uT"),
        "bias_final_error_norm_uT_p50": _column_quantile(
            summary_df, "bias_final_error_norm_uT", 0.5
        ),
        "bias_final_error_norm_uT_p95": _column_quantile(
            summary_df, "bias_final_error_norm_uT", 0.95
        ),
        "innovation_mean_norm_uT_mean": _column_mean(summary_df, "innovation_mean_norm_uT"),
        "nis_mean_mean": _column_mean(summary_df, "nis_mean"),
        "nis_central_95_fraction_mean": _column_mean(summary_df, "nis_central_95_fraction"),
        "attitude_settled_below_1_deg_fraction": _nonnull_fraction(
            summary_df, "attitude_settled_below_1_deg_s"
        ),
        "bias_settled_below_0p15_uT_fraction": _nonnull_fraction(
            summary_df, "bias_settled_below_0p15_uT_s"
        ),
    }

    for threshold in ATTITUDE_SETTLING_THRESHOLDS_DEG:
        label = _threshold_label(threshold)
        summary.update(
            _settling_summary(
                summary_df,
                f"attitude_settled_below_{label}_deg_s",
                f"attitude_settled_below_{label}_deg",
                period_s,
            )
        )

    return summary


def save_monte_carlo_plots(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save compact Monte Carlo distribution plots."""

    paths = []
    plot_specs = [
        (
            "monte_carlo_attitude_final.png",
            "attitude_final_deg",
            "Final attitude error [deg]",
            "Monte Carlo final attitude error",
        ),
        (
            "monte_carlo_bias_final.png",
            "bias_final_error_norm_uT",
            "Final bias error norm [uT]",
            "Monte Carlo final bias error",
        ),
        (
            "monte_carlo_nis_mean.png",
            "nis_mean",
            "Mean NIS [-]",
            "Monte Carlo innovation consistency",
        ),
    ]

    for filename, column, xlabel, title in plot_specs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(summary_df[column].dropna(), bins=min(10, len(summary_df)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Run count")
        ax.set_title(title)
        ax.grid(True)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    return paths


def write_validation_report(
    path: Path,
    config: AEKFValidationConfig,
    magnetometer_config: MagnetometerConfig,
    period_s: float,
    monte_carlo_summary: dict[str, Any],
    long_summary: dict[str, Any],
) -> None:
    """Write a short Markdown report for supervisor-facing validation."""

    rotation_rows = _format_matrix(magnetometer_config.sensor_axes_from_body)
    position_rows = _format_matrix(magnetometer_config.positions_body_m)
    mc_bias_p95 = monte_carlo_summary["bias_final_error_norm_uT_p95"]
    mc_nis_fraction = monte_carlo_summary["nis_central_95_fraction_mean"]
    mc_attitude_settling_rows = _format_attitude_settling_rows(monte_carlo_summary)
    long_bias_best = long_summary["bias_best_error_norm_uT"]
    long_bias_best_time = long_summary["bias_best_time_s"]
    report = f"""# AEKF Validation Summary

## Scope

This validation run covers four checks:

- Monte Carlo runs with different measurement-noise seeds and initial AEKF errors.
- Normalized innovation squared (NIS) consistency against chi-square bounds.
- A longer AEKF run covering {config.long_run_orbits:g} orbits.
- Current magnetometer geometry and limitations.

## Setup

- Estimated orbital period: {period_s:.3f} s.
- Monte Carlo runs: {config.monte_carlo_runs}.
- Initial attitude error range: 0 to {config.initial_attitude_error_max_deg:g} deg.
- Initial omega error std: {config.initial_omega_error_std_degps:g} deg/s per axis.
- Initial bias estimate std: {config.initial_bias_estimate_std_uT:g} uT per axis.

## Monte Carlo Aggregate

- Mean final attitude error: {monte_carlo_summary["attitude_final_deg_mean"]:.3f} deg.
- 95th percentile final attitude error: {monte_carlo_summary["attitude_final_deg_p95"]:.3f} deg.
- Mean final bias error norm: {monte_carlo_summary["bias_final_error_norm_uT_mean"]:.3f} uT.
- 95th percentile final bias error norm: {mc_bias_p95:.3f} uT.
- Mean NIS over runs: {monte_carlo_summary["nis_mean_mean"]:.3f}.
- Mean central-95% NIS fraction: {mc_nis_fraction:.3f}.

## Monte Carlo Attitude Settling

| Threshold | Settled runs | Mean [s] | Median [s] | P95 [s] | Mean [orbits] |
| --- | ---: | ---: | ---: | ---: | ---: |
{mc_attitude_settling_rows}

## Long Run

- Duration: {long_summary["duration_s"]:.1f} s ({long_summary["duration_orbits"]:.2f} orbits).
- Final attitude error: {long_summary["attitude_final_deg"]:.3f} deg.
- RMS attitude error: {long_summary["attitude_rms_deg"]:.3f} deg.
- Final bias error norm: {long_summary["bias_final_error_norm_uT"]:.3f} uT.
- Best bias error norm: {long_bias_best:.3f} uT at {long_bias_best_time:.1f} s.
- Mean NIS: {long_summary["nis_mean"]:.3f}.
- Central-95% NIS fraction: {long_summary["nis_central_95_fraction"]:.3f}.

## Magnetometer Model

The current measurement model is:

```text
B_sensor = C_sensor_from_body * R_eci_from_body(q).T * B_eci
           + bias_sensor + noise
```

The current sensor-axis matrix is:

```text
{rotation_rows}
```

The current sensor positions in body coordinates are:

```text
{position_rows}
```

The positions are stored and validated, but the current field model supplies one
uniform body-frame field vector at the spacecraft center. Therefore these small
translations do not yet alter the measurement unless a field-gradient or local
magnetic-disturbance model is added.
"""
    path.write_text(report, encoding="utf-8")


def _build_base_truth_result(
    elements: ClassicalOrbitalElements,
    simulation_config: SimulationConfig,
    attitude_config: AttitudeConfig,
    magnetometer_config: MagnetometerConfig,
) -> SimulationResult:
    runner = SimulationRunner(
        magnetometer_model=MagnetometerModel(
            bias_sensor_t=magnetometer_config.bias_sensor_t,
            noise_std_t=0.0,
            seed=None,
            sensor_axes_from_body=magnetometer_config.sensor_axes_from_body,
            positions_body_m=magnetometer_config.positions_body_m,
        ),
        kalman_filter=None,
    )

    return runner.run(elements, simulation_config, attitude_config)


def _estimate_aekf(
    base_result: SimulationResult,
    attitude_config: AttitudeConfig,
    magnetometer_config: MagnetometerConfig,
    measurement_sensor_t: ArrayFloat64,
    initial_quaternion: ArrayFloat64,
    initial_omega_radps: ArrayFloat64,
    initial_bias_sensor_t: ArrayFloat64,
) -> KalmanFilterEstimate:
    aekf = AEKF(
        AEKFConfig(
            initial_quaternion_eci_from_body=initial_quaternion,
            initial_omega_body_radps=initial_omega_radps,
            initial_magnetometer_bias_sensor_t=initial_bias_sensor_t,
            inertia_kg_m2=attitude_config.inertia_kg_m2,
            torque_body_nm=attitude_config.torque_body_nm,
            sensor_axes_from_body=magnetometer_config.sensor_axes_from_body,
            measurement_noise=_measurement_noise_from_sensor_config(magnetometer_config),
        )
    )

    return aekf.estimate(
        KalmanFilterInput(
            t_s=base_result.orbit.t_s,
            measurements_body_t=measurement_sensor_t,
            reference_vectors_eci_t=base_result.magnetic_field.b_eci_t,
        )
    )


def _measurement_noise_from_sensor_config(config: MagnetometerConfig) -> np.ndarray:
    noise_std_t = np.asarray(config.noise_std_t, dtype=np.float64)

    if noise_std_t.shape == ():
        return np.eye(3, dtype=np.float64) * float(noise_std_t) ** 2

    return np.diag(noise_std_t**2)


def _perturb_quaternion(
    quaternion: ArrayFloat64, rng: np.random.Generator, angle_rad: float
) -> ArrayFloat64:
    axis = _random_unit_vector(rng)
    half_angle = 0.5 * angle_rad
    delta = np.array([np.cos(half_angle), *(np.sin(half_angle) * axis)], dtype=np.float64)

    return normalize_quaternion(quaternion_multiply(delta, quaternion))


def _random_unit_vector(rng: np.random.Generator) -> ArrayFloat64:
    vector = rng.normal(size=3)
    norm = float(np.linalg.norm(vector))

    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    return np.asarray(vector / norm, dtype=np.float64)


def _settling_time_s(times_s: ArrayFloat64, values: ArrayFloat64, threshold: float) -> float | None:
    violations = np.asarray(values, dtype=np.float64) > threshold
    future_violations = np.maximum.accumulate(violations[::-1])[::-1]
    indices = np.where(~future_violations)[0]

    if len(indices) == 0:
        return None

    return float(times_s[int(indices[0])])


def _threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _column_mean(df: pd.DataFrame, column: str) -> float:
    return float(df[column].mean())


def _column_quantile(df: pd.DataFrame, column: str, quantile: float) -> float:
    return float(df[column].quantile(quantile))


def _nonnull_fraction(df: pd.DataFrame, column: str) -> float:
    return float(df[column].notna().mean())


def _settling_summary(
    df: pd.DataFrame, column: str, prefix: str, period_s: float | None
) -> dict[str, Any]:
    settled_times_s = df[column].dropna()
    summary: dict[str, Any] = {
        f"{prefix}_count": int(len(settled_times_s)),
        f"{prefix}_fraction": float(len(settled_times_s) / len(df)),
        f"{prefix}_mean_s": None,
        f"{prefix}_median_s": None,
        f"{prefix}_p95_s": None,
        f"{prefix}_mean_orbits": None,
        f"{prefix}_median_orbits": None,
    }

    if len(settled_times_s) == 0:
        return summary

    mean_s = float(settled_times_s.mean())
    median_s = float(settled_times_s.median())
    summary.update(
        {
            f"{prefix}_mean_s": mean_s,
            f"{prefix}_median_s": median_s,
            f"{prefix}_p95_s": float(settled_times_s.quantile(0.95)),
        }
    )

    if period_s is not None:
        summary.update(
            {
                f"{prefix}_mean_orbits": mean_s / period_s,
                f"{prefix}_median_orbits": median_s / period_s,
            }
        )

    return summary


def _format_attitude_settling_rows(summary: dict[str, Any]) -> str:
    rows = []

    for threshold in ATTITUDE_SETTLING_THRESHOLDS_DEG:
        label = _threshold_label(threshold)
        prefix = f"attitude_settled_below_{label}_deg"
        rows.append(
            "| "
            f"< {threshold:g} deg | "
            f"{summary[f'{prefix}_count']}/{summary['runs']} | "
            f"{_format_optional_float(summary[f'{prefix}_mean_s'])} | "
            f"{_format_optional_float(summary[f'{prefix}_median_s'])} | "
            f"{_format_optional_float(summary[f'{prefix}_p95_s'])} | "
            f"{_format_optional_float(summary[f'{prefix}_mean_orbits'], precision=2)} |"
        )

    return "\n".join(rows)


def _format_optional_float(value: Any, precision: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"

    return f"{float(value):.{precision}f}"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _format_matrix(matrix: ArrayFloat64) -> str:
    return "\n".join(
        "[" + ", ".join(f"{value: .6f}" for value in np.asarray(row, dtype=np.float64)) + "]"
        for row in matrix
    )


def main() -> None:
    """Command-line entry point for ``hatch run validate-aekf``."""

    import argparse

    parser = argparse.ArgumentParser(description="Run AEKF validation artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VALIDATION_OUTPUT_DIR)
    parser.add_argument(
        "--monte-carlo-runs", type=int, default=DEFAULT_VALIDATION_CONFIG.monte_carlo_runs
    )
    parser.add_argument(
        "--long-orbits", type=float, default=DEFAULT_VALIDATION_CONFIG.long_run_orbits
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_VALIDATION_CONFIG.random_seed)
    args = parser.parse_args()

    artifacts = run_aekf_validation(
        args.output_dir,
        AEKFValidationConfig(
            monte_carlo_runs=args.monte_carlo_runs,
            random_seed=args.random_seed,
            long_run_orbits=args.long_orbits,
        ),
    )

    for label, path in artifacts.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
