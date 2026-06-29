"""Export orbit results to CesiumJS-compatible CZML visualisations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

from simulation.attitude import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from simulation.frames import _as_time_array

REQUIRED_COLUMNS = ("t_utc", "x_ecef_m", "y_ecef_m", "z_ecef_m", "lat_deg", "lon_deg", "alt_m")
ORIENTATION_COLUMNS = (
    "x_eci_m",
    "y_eci_m",
    "z_eci_m",
    "q_eci_from_body_w",
    "q_eci_from_body_x",
    "q_eci_from_body_y",
    "q_eci_from_body_z",
)

VIEWER_TEMPLATE_PATH = Path(__file__).resolve().with_name("index.html")


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a helpful error if a DataFrame is missing required columns."""

    missing_columns = set(columns).difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns required for Cesium visualisation: {missing}")


def _parse_utc_timestamps(df: pd.DataFrame) -> pd.Series:
    """Parse the UTC timestamp column into timezone-aware pandas timestamps."""

    timestamps = pd.Series(pd.to_datetime(df["t_utc"].to_numpy(), utc=True, errors="raise"))

    if timestamps.empty:
        raise ValueError("Cannot build a Cesium visualisation from an empty DataFrame.")

    if timestamps.isna().any():
        raise ValueError("The t_utc column contains invalid timestamps.")

    if not timestamps.is_monotonic_increasing:
        raise ValueError("The t_utc column must be monotonically increasing.")

    return timestamps


def _format_czml_time(timestamp: pd.Timestamp) -> str:
    """Format a timestamp as an ISO-8601 UTC string accepted by CZML."""

    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _sample_seconds(timestamps: pd.Series) -> list[float]:
    """Return seconds elapsed since the first timestamp."""

    elapsed = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    return [float(value) for value in elapsed]


def _sampled_cartesian_positions(df: pd.DataFrame, seconds: list[float]) -> list[float]:
    """Build CZML sampled cartesian position values in the fixed frame."""

    positions: list[float] = []

    for elapsed_s, (_, row) in zip(seconds, df.iterrows(), strict=True):
        positions.extend(
            [elapsed_s, float(row["x_ecef_m"]), float(row["y_ecef_m"]), float(row["z_ecef_m"])]
        )

    return positions


def _static_cartesian_positions(df: pd.DataFrame) -> list[float]:
    """Build static CZML cartesian position list values."""

    positions: list[float] = []

    for _, row in df.iterrows():
        positions.extend([float(row["x_ecef_m"]), float(row["y_ecef_m"]), float(row["z_ecef_m"])])

    return positions


def _ground_track_positions(df: pd.DataFrame) -> list[float]:
    """Build lon/lat/height triples for a CZML ground-track polyline."""

    positions: list[float] = []

    for _, row in df.iterrows():
        positions.extend([float(row["lon_deg"]), float(row["lat_deg"]), 0.0])

    return positions


def _cartesian_to_array(cartesian) -> np.ndarray:
    """Return an astropy cartesian representation as a meter-valued vector."""

    return np.array(
        [cartesian.x.to_value(u.m), cartesian.y.to_value(u.m), cartesian.z.to_value(u.m)],
        dtype=np.float64,
    )


def _orthonormalized_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""

    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right

    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right

    return rotation


def _fixed_from_eci_rotation_matrices(df: pd.DataFrame) -> np.ndarray:
    """Estimate local ECEF/FIXED-from-ECI rotation matrices for each timestamp."""

    times = _as_time_array(df["t_utc"].to_numpy())
    r_eci_m = df[["x_eci_m", "y_eci_m", "z_eci_m"]].to_numpy(dtype=np.float64)
    basis_eci = np.eye(3, dtype=np.float64)
    rotations = np.empty((len(df), 3, 3), dtype=np.float64)

    for idx, (time, position_eci) in enumerate(zip(times, r_eci_m, strict=True)):
        base_gcrs = GCRS(
            CartesianRepresentation(
                position_eci[0] * u.m, position_eci[1] * u.m, position_eci[2] * u.m
            ),
            obstime=time,
        )
        base_fixed = _cartesian_to_array(base_gcrs.transform_to(ITRS(obstime=time)).cartesian)

        rotation = np.empty((3, 3), dtype=np.float64)

        for col, basis_vector in enumerate(basis_eci):
            shifted_position = position_eci + basis_vector
            shifted_gcrs = GCRS(
                CartesianRepresentation(
                    shifted_position[0] * u.m, shifted_position[1] * u.m, shifted_position[2] * u.m
                ),
                obstime=time,
            )
            shifted_fixed = _cartesian_to_array(
                shifted_gcrs.transform_to(ITRS(obstime=time)).cartesian
            )
            rotation[:, col] = shifted_fixed - base_fixed

        rotations[idx] = _orthonormalized_rotation(rotation)

    return rotations


def _sampled_body_orientation(df: pd.DataFrame, seconds: list[float]) -> list[float] | None:
    """Build sampled CZML quaternions for the body-to-fixed orientation."""

    if not set(ORIENTATION_COLUMNS).issubset(df.columns):
        return None

    fixed_from_eci = _fixed_from_eci_rotation_matrices(df)
    quaternions = df[
        ["q_eci_from_body_w", "q_eci_from_body_x", "q_eci_from_body_y", "q_eci_from_body_z"]
    ].to_numpy(dtype=np.float64)
    orientation: list[float] = []

    for elapsed_s, rotation_fixed_from_eci, quaternion_eci_from_body in zip(
        seconds, fixed_from_eci, quaternions, strict=True
    ):
        rotation_eci_from_body = quaternion_to_rotation_matrix(quaternion_eci_from_body)
        rotation_fixed_from_body = rotation_fixed_from_eci @ rotation_eci_from_body
        quaternion_fixed_from_body = rotation_matrix_to_quaternion(rotation_fixed_from_body)
        qw, qx, qy, qz = quaternion_fixed_from_body
        orientation.extend([elapsed_s, qx, qy, qz, qw])

    return orientation


def dataframe_to_czml(df: pd.DataFrame, clock_multiplier: float = 60.0) -> list[dict]:
    """
    Convert an orbit simulation DataFrame to a CZML document.

    Positions are exported in the Earth-fixed frame so the visualisation follows
    the same Earth-relative trajectory as the geodetic and ECEF outputs.
    """

    _require_columns(df, REQUIRED_COLUMNS)

    timestamps = _parse_utc_timestamps(df)
    sample_seconds = _sample_seconds(timestamps)
    start_time = _format_czml_time(timestamps.iloc[0])
    stop_time = _format_czml_time(timestamps.iloc[-1])
    availability = f"{start_time}/{stop_time}"
    duration_s = max(sample_seconds[-1], 1.0)
    body_orientation = _sampled_body_orientation(df, sample_seconds)

    first_row = df.iloc[0]
    altitude_km = float(df["alt_m"].mean()) / 1000.0
    radius_km = (
        (
            df["x_ecef_m"].to_numpy() ** 2
            + df["y_ecef_m"].to_numpy() ** 2
            + df["z_ecef_m"].to_numpy() ** 2
        )
        ** 0.5
    ).mean() / 1000.0

    satellite_packet = {
        "id": "cubesat",
        "name": "CubeSat",
        "availability": availability,
        "description": (
            "<table>"
            f"<tr><th>Start UTC</th><td>{start_time}</td></tr>"
            f"<tr><th>Mean altitude</th><td>{altitude_km:.1f} km</td></tr>"
            f"<tr><th>Mean radius</th><td>{radius_km:.1f} km</td></tr>"
            f"<tr><th>Initial latitude</th><td>{float(first_row['lat_deg']):.3f} deg</td></tr>"
            f"<tr><th>Initial longitude</th><td>{float(first_row['lon_deg']):.3f} deg</td></tr>"
            "</table>"
        ),
        "position": {
            "epoch": start_time,
            "referenceFrame": "FIXED",
            "interpolationAlgorithm": "LAGRANGE",
            "interpolationDegree": 3,
            "cartesian": _sampled_cartesian_positions(df, sample_seconds),
        },
        "point": {
            "pixelSize": 10,
            "color": {"rgba": [255, 214, 102, 255]},
            "outlineColor": {"rgba": [12, 18, 28, 255]},
            "outlineWidth": 2,
            "scaleByDistance": {"nearFarScalar": [1000000.0, 1.8, 12000000.0, 0.7]},
        },
        "box": {
            "dimensions": {"cartesian": [120000.0, 70000.0, 45000.0]},
            "material": {"solidColor": {"color": {"rgba": [255, 214, 102, 95]}}},
            "outline": True,
            "outlineColor": {"rgba": [255, 255, 255, 210]},
        },
        "label": {
            "text": "CubeSat",
            "font": "13px sans-serif",
            "fillColor": {"rgba": [242, 247, 255, 245]},
            "outlineColor": {"rgba": [12, 18, 28, 255]},
            "outlineWidth": 3,
            "style": "FILL_AND_OUTLINE",
            "pixelOffset": {"cartesian2": [0, -24]},
            "scaleByDistance": {"nearFarScalar": [1000000.0, 1.0, 11000000.0, 0.0]},
        },
        "path": {
            "show": [{"interval": availability, "boolean": True}],
            "width": 3,
            "leadTime": 0,
            "trailTime": duration_s,
            "resolution": 30,
            "material": {
                "polylineGlow": {"color": {"rgba": [73, 190, 255, 230]}, "glowPower": 0.16}
            },
        },
    }

    if body_orientation is not None:
        satellite_packet["orientation"] = {
            "epoch": start_time,
            "interpolationAlgorithm": "LINEAR",
            "interpolationDegree": 1,
            "unitQuaternion": body_orientation,
        }

    return [
        {
            "id": "document",
            "name": "CubeSat Cesium visualisation",
            "version": "1.0",
            "clock": {
                "interval": availability,
                "currentTime": start_time,
                "multiplier": float(clock_multiplier),
                "range": "LOOP_STOP",
                "step": "SYSTEM_CLOCK_MULTIPLIER",
            },
        },
        satellite_packet,
        {
            "id": "orbit-outline",
            "name": "Orbit outline",
            "availability": availability,
            "polyline": {
                "positions": {"cartesian": _static_cartesian_positions(df)},
                "width": 2,
                "material": {
                    "polylineGlow": {"color": {"rgba": [92, 230, 180, 170]}, "glowPower": 0.12}
                },
            },
        },
        {
            "id": "ground-track",
            "name": "Ground track",
            "availability": availability,
            "polyline": {
                "positions": {"cartographicDegrees": _ground_track_positions(df)},
                "clampToGround": True,
                "width": 2,
                "material": {
                    "polylineDash": {"color": {"rgba": [255, 255, 255, 165]}, "dashLength": 12}
                },
            },
        },
    ]


def write_czml(df: pd.DataFrame, output_path: Path, clock_multiplier: float = 60.0) -> Path:
    """Write a DataFrame as a CZML document and return the written path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    czml = dataframe_to_czml(df, clock_multiplier=clock_multiplier)
    output_path.write_text(json.dumps(czml, indent=2), encoding="utf-8")

    return output_path


def write_viewer(output_path: Path) -> Path:
    """Copy the static CesiumJS viewer next to the generated CZML file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(VIEWER_TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    return output_path


def create_cesium_visualisation(
    df: pd.DataFrame, output_dir: Path, clock_multiplier: float = 60.0
) -> tuple[Path, Path]:
    """Create the CesiumJS viewer and CZML output files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    czml_path = write_czml(df, output_dir / "orbit.czml", clock_multiplier=clock_multiplier)
    viewer_path = write_viewer(output_dir / "index.html")

    return czml_path, viewer_path
