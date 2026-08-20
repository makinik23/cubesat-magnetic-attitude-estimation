"""Visualize Poinsot's construction for torque-free rigid-body motion.

Run from the repository root with:

    python learning/poinsot_construction.py

or from the ``learning`` directory with:

    python poinsot_construction.py

The left panel shows the body-frame polhode on the fixed normalized energy ellipsoid.
The right panel shows the inertial-frame herpolhode on the fixed invariable
plane, with the same ellipsoid rotating tangent to that plane.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Array = np.ndarray


@dataclass(frozen=True)
class PoinsotConfig:
    inertia: Array
    omega0_body: Array
    duration_s: float
    steps: int
    trail_length: int
    substeps_per_frame: int


@dataclass(frozen=True)
class PoinsotSolution:
    times_s: Array
    omega_body: Array
    h_body: Array
    rotation_inertial_from_body: Array
    contact_body: Array
    contact_inertial: Array
    energy: float
    h_inertial: Array


@dataclass(frozen=True)
class PoinsotScene:
    fig: plt.Figure
    update_frame: Callable[[int], Any]
    frame_count: int


def quaternion_multiply(left: Array, right: Array) -> Array:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right

    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def quaternion_to_rotation_matrix(quaternion: Array) -> Array:
    qw, qx, qy, qz = quaternion / np.linalg.norm(quaternion)

    return np.array(
        [
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx**2 + qy**2)],
        ],
        dtype=float,
    )


def torque_free_derivative(state: Array, inertia: Array) -> Array:
    quaternion = state[:4] / np.linalg.norm(state[:4])
    omega = state[4:]

    h_body = inertia * omega
    omega_dot = -np.cross(omega, h_body) / inertia
    quaternion_dot = 0.5 * quaternion_multiply(quaternion, np.array([0.0, *omega]))

    return np.concatenate((quaternion_dot, omega_dot))


def rk4_step(state: Array, dt_s: float, inertia: Array) -> Array:
    k1 = torque_free_derivative(state, inertia)
    k2 = torque_free_derivative(state + 0.5 * dt_s * k1, inertia)
    k3 = torque_free_derivative(state + 0.5 * dt_s * k2, inertia)
    k4 = torque_free_derivative(state + dt_s * k3, inertia)

    next_state = state + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    next_state[:4] /= np.linalg.norm(next_state[:4])

    return next_state


def propagate_poinsot(config: PoinsotConfig) -> PoinsotSolution:
    if config.steps < 2:
        raise ValueError("steps must be at least 2")
    if config.substeps_per_frame < 1:
        raise ValueError("substeps_per_frame must be at least 1")

    inertia = np.asarray(config.inertia, dtype=float)
    omega0_body = np.asarray(config.omega0_body, dtype=float)

    if inertia.shape != (3,) or omega0_body.shape != (3,):
        raise ValueError("inertia and omega0_body must be three-element vectors")
    if np.any(inertia <= 0.0):
        raise ValueError("principal moments of inertia must be positive")

    times_s = np.linspace(0.0, config.duration_s, config.steps)
    frame_dt_s = times_s[1] - times_s[0]
    dt_s = frame_dt_s / config.substeps_per_frame
    states = np.zeros((config.steps, 7), dtype=float)
    states[0, :4] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    states[0, 4:] = omega0_body

    for index in range(1, config.steps):
        state = states[index - 1]
        for _ in range(config.substeps_per_frame):
            state = rk4_step(state, dt_s, inertia)
        states[index] = state

    omega_body = states[:, 4:]
    h_body = omega_body * inertia[np.newaxis, :]
    rotations = np.asarray([quaternion_to_rotation_matrix(q) for q in states[:, :4]], dtype=float)

    energy = float(0.5 * np.dot(inertia * omega0_body, omega0_body))
    contact_body = omega_body / np.sqrt(2.0 * energy)
    contact_inertial = np.einsum("nij,nj->ni", rotations, contact_body)
    h_inertial = rotations[0] @ h_body[0]

    return PoinsotSolution(
        times_s=times_s,
        omega_body=omega_body,
        h_body=h_body,
        rotation_inertial_from_body=rotations,
        contact_body=contact_body,
        contact_inertial=contact_inertial,
        energy=energy,
        h_inertial=h_inertial,
    )


def energy_ellipsoid_mesh(
    inertia: Array, n_u: int = 42, n_v: int = 22
) -> tuple[Array, Array, Array]:
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    axes = 1.0 / np.sqrt(inertia)

    x = axes[0] * np.cos(uu) * np.sin(vv)
    y = axes[1] * np.sin(uu) * np.sin(vv)
    z = axes[2] * np.cos(vv)

    return x, y, z


def rotate_mesh(rotation: Array, mesh: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
    x, y, z = mesh
    points = np.stack((x.ravel(), y.ravel(), z.ravel()))
    rotated = rotation @ points

    return (rotated[0].reshape(x.shape), rotated[1].reshape(y.shape), rotated[2].reshape(z.shape))


def plane_basis(normal: Array) -> tuple[Array, Array, Array]:
    unit_normal = normal / np.linalg.norm(normal)
    seed = np.array([1.0, 0.0, 0.0], dtype=float)

    if abs(np.dot(seed, unit_normal)) > 0.85:
        seed = np.array([0.0, 1.0, 0.0], dtype=float)

    first = np.cross(unit_normal, seed)
    first /= np.linalg.norm(first)
    second = np.cross(unit_normal, first)

    return unit_normal, first, second


def plane_patch(normal: Array, center: Array, side_length: float) -> Array:
    _, first, second = plane_basis(normal)
    half = 0.5 * side_length

    return np.array(
        [
            center - half * first - half * second,
            center + half * first - half * second,
            center + half * first + half * second,
            center - half * first + half * second,
        ],
        dtype=float,
    )


def invariable_plane_patch(solution: PoinsotSolution, side_length: float) -> Array:
    normal, _, _ = plane_basis(solution.h_inertial)
    distance = np.sqrt(2.0 * solution.energy) / np.linalg.norm(solution.h_inertial)
    center = normal * distance

    return plane_patch(solution.h_inertial, center, side_length)


def set_axes_equal(ax: Any, radius: float) -> None:
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def draw_arrow(
    ax: Any, tail: Array, vector: Array, label: str, color: str, *, length: float | None = None
) -> tuple[Any, Any]:
    displayed = np.asarray(vector, dtype=float)

    if length is not None:
        displayed = length * displayed / np.linalg.norm(displayed)

    arrow = ax.quiver(
        tail[0],
        tail[1],
        tail[2],
        displayed[0],
        displayed[1],
        displayed[2],
        color=color,
        linewidth=1.8,
        arrow_length_ratio=0.12,
    )
    text_position = tail + 1.08 * displayed
    text = ax.text(*text_position, label, color=color)

    return arrow, text


def configure_axes(
    ax_body: Any, ax_inertial: Any, radius: float, solution: PoinsotSolution, plane_vertices: Array
) -> None:
    ax_body.set_title("Body frame: fixed energy ellipsoid and polhode")
    ax_body.set_xlabel("b1")
    ax_body.set_ylabel("b2")
    ax_body.set_zlabel("b3")
    set_axes_equal(ax_body, radius)
    ax_body.view_init(elev=24.0, azim=-50.0)

    ax_inertial.set_title("Inertial frame: rolling ellipsoid and herpolhode")
    ax_inertial.set_xlabel("i1")
    ax_inertial.set_ylabel("i2")
    ax_inertial.set_zlabel("i3")
    set_axes_equal(ax_inertial, radius)
    ax_inertial.view_init(elev=22.0, azim=-55.0)

    draw_arrow(
        ax_inertial,
        np.zeros(3),
        solution.h_inertial,
        "H inertial direction",
        "#1f77b4",
        length=0.78 * radius,
    )

    plane = Poly3DCollection([plane_vertices], alpha=0.18, facecolor="#87aade", edgecolor="#5874a6")
    ax_inertial.add_collection3d(plane)


def build_poinsot_scene(config: PoinsotConfig) -> PoinsotScene:
    solution = propagate_poinsot(config)
    mesh_body = energy_ellipsoid_mesh(config.inertia)
    axes = 1.0 / np.sqrt(config.inertia)
    radius = 1.7 * max(np.max(axes), np.max(np.linalg.norm(solution.contact_inertial, axis=1)))
    plane_vertices = invariable_plane_patch(solution, side_length=1.75 * radius)

    fig = plt.figure(figsize=(13.0, 6.5))
    ax_body = fig.add_subplot(1, 2, 1, projection="3d")
    ax_inertial = fig.add_subplot(1, 2, 2, projection="3d")
    configure_axes(ax_body, ax_inertial, radius, solution, plane_vertices)

    ax_body.plot_surface(*mesh_body, color="#d7d7d7", alpha=0.22, linewidth=0.0)
    ax_body.plot_wireframe(
        *mesh_body, color="#777777", alpha=0.24, linewidth=0.35, rstride=3, cstride=3
    )
    ax_body.plot(
        solution.contact_body[:, 0],
        solution.contact_body[:, 1],
        solution.contact_body[:, 2],
        color="#111111",
        linewidth=1.1,
        alpha=0.38,
    )
    ax_inertial.plot(
        solution.contact_inertial[:, 0],
        solution.contact_inertial[:, 1],
        solution.contact_inertial[:, 2],
        color="#111111",
        linewidth=1.1,
        alpha=0.38,
    )

    h_body_norm = np.linalg.norm(solution.h_body, axis=1)
    h_body_norm0 = h_body_norm[0]
    energy = 0.5 * np.sum(config.inertia[np.newaxis, :] * solution.omega_body**2, axis=1)
    energy0 = energy[0]
    omega_inertial = np.einsum(
        "nij,nj->ni", solution.rotation_inertial_from_body, solution.omega_body
    )
    ellipsoid_value = np.sum(config.inertia[np.newaxis, :] * solution.contact_body**2, axis=1)
    plane_value = solution.contact_inertial @ solution.h_inertial / np.sqrt(2.0 * solution.energy)
    slip_speed = np.linalg.norm(np.cross(omega_inertial, solution.contact_inertial), axis=1)

    (body_trail,) = ax_body.plot([], [], [], color="#d62728", linewidth=2.5)
    (inertial_trail,) = ax_inertial.plot([], [], [], color="#d62728", linewidth=2.5)
    (body_point,) = ax_body.plot([], [], [], "o", color="#d62728", markersize=7.0)
    (inertial_point,) = ax_inertial.plot([], [], [], "o", color="#d62728", markersize=7.0)
    inertial_wireframe: list[Any | None] = [None]
    body_tangent_plane: list[Any | None] = [None]
    body_vector_artists: list[tuple[Any, Any]] = []
    inertial_vector_artists: list[tuple[Any, Any]] = []

    time_text = fig.text(0.5, 0.035, "", ha="center", va="center", fontsize=9)
    fig.suptitle("Poinsot construction for torque-free rigid-body motion", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))

    def update(frame: int) -> tuple:
        start = max(0, frame - config.trail_length)
        body_segment = solution.contact_body[start : frame + 1]
        inertial_segment = solution.contact_inertial[start : frame + 1]

        body_trail.set_data(body_segment[:, 0], body_segment[:, 1])
        body_trail.set_3d_properties(body_segment[:, 2])
        inertial_trail.set_data(inertial_segment[:, 0], inertial_segment[:, 1])
        inertial_trail.set_3d_properties(inertial_segment[:, 2])

        body_point.set_data([solution.contact_body[frame, 0]], [solution.contact_body[frame, 1]])
        body_point.set_3d_properties([solution.contact_body[frame, 2]])
        inertial_point.set_data(
            [solution.contact_inertial[frame, 0]], [solution.contact_inertial[frame, 1]]
        )
        inertial_point.set_3d_properties([solution.contact_inertial[frame, 2]])

        if body_tangent_plane[0] is not None:
            body_tangent_plane[0].remove()

        body_plane_vertices = plane_patch(
            solution.h_body[frame], solution.contact_body[frame], side_length=0.95 * radius
        )
        body_tangent_plane[0] = Poly3DCollection(
            [body_plane_vertices], alpha=0.16, facecolor="#87aade", edgecolor="#5874a6"
        )
        ax_body.add_collection3d(body_tangent_plane[0])

        for arrow, text in body_vector_artists:
            arrow.remove()
            text.remove()
        body_vector_artists.clear()
        body_vector_artists.append(
            draw_arrow(ax_body, np.zeros(3), solution.contact_body[frame], "r_B", "#d62728")
        )
        body_vector_artists.append(
            draw_arrow(
                ax_body,
                solution.contact_body[frame],
                solution.h_body[frame],
                "normal || H_B",
                "#1f77b4",
                length=0.25 * radius,
            )
        )

        for arrow, text in inertial_vector_artists:
            arrow.remove()
            text.remove()
        inertial_vector_artists.clear()
        inertial_vector_artists.append(
            draw_arrow(ax_inertial, np.zeros(3), solution.contact_inertial[frame], "r_I", "#d62728")
        )

        if inertial_wireframe[0] is not None:
            inertial_wireframe[0].remove()

        rotated_mesh = rotate_mesh(solution.rotation_inertial_from_body[frame], mesh_body)
        inertial_wireframe[0] = ax_inertial.plot_wireframe(
            *rotated_mesh, color="#666666", alpha=0.34, linewidth=0.35, rstride=3, cstride=3
        )

        time_text.set_text(
            f"t = {solution.times_s[frame]:.2f} s   "
            f"|H|/|H0| = {h_body_norm[frame] / h_body_norm0:.9f}   "
            f"T/T0 = {energy[frame] / energy0:.9f}\n"
            f"r^T I r = {ellipsoid_value[frame]:.9f}   "
            f"H.r/sqrt(2T0) = {plane_value[frame]:.9f}   "
            f"|omega x r| = {slip_speed[frame]:.2e}"
        )

        return body_trail, inertial_trail, body_point, inertial_point, time_text

    update(0)

    return PoinsotScene(fig=fig, update_frame=update, frame_count=config.steps)


def build_animation(
    config: PoinsotConfig, interval_ms: int
) -> tuple[plt.Figure, animation.FuncAnimation]:
    scene = build_poinsot_scene(config)
    anim = animation.FuncAnimation(
        scene.fig, scene.update_frame, frames=scene.frame_count, interval=interval_ms, blit=False
    )

    return scene.fig, anim


def save_animation(anim: animation.FuncAnimation, output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    writer: Any

    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix in {".mp4", ".m4v"}:
        writer = animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError("animation output must end with .gif, .mp4, or .m4v")

    anim.save(output, writer=writer, dpi=140)


def save_static_snapshot(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=55.0, help="simulation duration in seconds"
    )
    parser.add_argument("--steps", type=int, default=720, help="number of animation frames")
    parser.add_argument(
        "--trail-length", type=int, default=160, help="number of frames in the red trail"
    )
    parser.add_argument(
        "--substeps-per-frame",
        type=int,
        default=8,
        help="internal RK4 substeps between rendered frames",
    )
    parser.add_argument("--fps", type=int, default=30, help="saved animation frames per second")
    parser.add_argument(
        "--interval-ms", type=int, default=35, help="interactive animation interval"
    )
    parser.add_argument(
        "--inertia",
        type=float,
        nargs=3,
        default=(2.0, 3.2, 5.0),
        metavar=("I1", "I2", "I3"),
        help="principal moments of inertia",
    )
    parser.add_argument(
        "--omega0",
        type=float,
        nargs=3,
        default=(0.72, 1.15, 1.62),
        metavar=("W1", "W2", "W3"),
        help="initial angular velocity in body coordinates",
    )
    parser.add_argument("--output", type=Path, help="optional .gif or .mp4 animation output")
    parser.add_argument("--static-output", type=Path, help="optional PNG/JPG snapshot output")
    parser.add_argument("--no-show", action="store_true", help="do not open an interactive window")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PoinsotConfig(
        inertia=np.asarray(args.inertia, dtype=float),
        omega0_body=np.asarray(args.omega0, dtype=float),
        duration_s=args.duration,
        steps=args.steps,
        trail_length=args.trail_length,
        substeps_per_frame=args.substeps_per_frame,
    )

    needs_animation = args.output is not None or not args.no_show

    if needs_animation:
        fig, anim = build_animation(config, interval_ms=args.interval_ms)
    else:
        scene = build_poinsot_scene(config)
        fig = scene.fig
        anim = None

    if args.static_output is not None:
        save_static_snapshot(fig, args.static_output)

    if args.output is not None:
        if anim is None:
            raise RuntimeError("animation output requested, but animation was not created")
        save_animation(anim, args.output, fps=args.fps)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
