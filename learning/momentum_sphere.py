import numpy as np
import matplotlib.pyplot as plt


def kinetic_energy(H, J):
    H1, H2, H3 = H
    J1, J2, J3 = J

    return 0.5 * (H1**2 / J1 + H2**2 / J2 + H3**2 / J3)


def sphere_coordinates(Hmag, n_theta=300, n_phi=600):
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)

    Phi, Theta = np.meshgrid(phi, theta)

    H1 = Hmag * np.sin(Theta) * np.cos(Phi)
    H2 = Hmag * np.sin(Theta) * np.sin(Phi)
    H3 = Hmag * np.cos(Theta)

    return Phi, Theta, H1, H2, H3


def energy_grid_on_sphere(H1, H2, H3, J):
    J1, J2, J3 = J

    E = 0.5 * (H1**2 / J1 + H2**2 / J2 + H3**2 / J3)

    return E


def strictly_increasing_levels(levels):
    levels = np.sort(np.asarray(levels, dtype=float))

    if len(levels) < 2:
        return levels

    energy_span = max(1.0, levels[-1] - levels[0])
    keep = np.concatenate(([True], np.diff(levels) > 1e-12 * energy_span))

    return levels[keep]


def get_contour_segments(Phi, Theta, E, levels):
    """
    Robimy kontury w przestrzeni parametrów sfery: phi-theta,
    a potem mapujemy je z powrotem na 3D.
    """

    fig_tmp, ax_tmp = plt.subplots()
    contour_set = ax_tmp.contour(Phi, Theta, E, levels=levels)
    plt.close(fig_tmp)

    return contour_set.allsegs


def map_angles_to_sphere(phi, theta, Hmag):
    H1 = Hmag * np.sin(theta) * np.cos(phi)
    H2 = Hmag * np.sin(theta) * np.sin(phi)
    H3 = Hmag * np.cos(theta)

    return H1, H2, H3


def plot_energy_paths_on_momentum_sphere(
    J, H0=None, Hmag=1.0, n_levels=30, title=None, show_selected_initial_condition=True
):
    """
    Rysuje sferę momentu pędu oraz krzywe stałej energii kinetycznej.

    J  = [J1, J2, J3]
    H0 = początkowy moment pędu [H1, H2, H3]

    Jeśli podasz H0, to promień sfery będzie równy |H0|,
    a dodatkowo zostanie pogrubiona krzywa energii przechodząca przez H0.
    """

    J = np.array(J, dtype=float)

    if H0 is not None:
        H0 = np.array(H0, dtype=float)
        Hmag = np.linalg.norm(H0)

    Phi, Theta, H1, H2, H3 = sphere_coordinates(Hmag)
    E = energy_grid_on_sphere(H1, H2, H3, J)

    Emin = Hmag**2 / (2 * np.max(J))
    Emax = Hmag**2 / (2 * np.min(J))

    eps = 1e-6 * (Emax - Emin)

    levels = np.linspace(Emin + eps, Emax - eps, n_levels)

    # Jeżeli ciało jest trójosiowe, dodajemy poziom energii dla osi pośredniej.
    # To pomaga zobaczyć separatrysy przy niestabilnej osi.
    if len(np.unique(J)) == 3:
        J_mid = np.sort(J)[1]
        E_mid = Hmag**2 / (2 * J_mid)
        levels = strictly_increasing_levels(np.append(levels, E_mid))

    contour_segments = get_contour_segments(Phi, Theta, E, levels)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Sfera momentu pędu
    ax.plot_surface(H1, H2, H3, alpha=0.12, linewidth=0, color="lightgray")

    ax.plot_wireframe(H1, H2, H3, rstride=20, cstride=30, linewidth=0.3, alpha=0.25, color="gray")

    # Krzywe stałej energii
    for level_segments in contour_segments:
        for segment in level_segments:
            if len(segment) < 2:
                continue

            phi_segment = segment[:, 0]
            theta_segment = segment[:, 1]

            x, y, z = map_angles_to_sphere(phi_segment, theta_segment, Hmag)

            ax.plot(x, y, z, color="black", linewidth=1.0)

    # Pogrubiona krzywa przechodząca przez zadany warunek początkowy
    if H0 is not None and show_selected_initial_condition:
        E0 = kinetic_energy(H0, J)

        selected_segments = get_contour_segments(Phi, Theta, E, [E0])

        for level_segments in selected_segments:
            for segment in level_segments:
                if len(segment) < 2:
                    continue

                phi_segment = segment[:, 0]
                theta_segment = segment[:, 1]

                x, y, z = map_angles_to_sphere(phi_segment, theta_segment, Hmag)

                ax.plot(x, y, z, color="red", linewidth=3.0)

        ax.scatter(H0[0], H0[1], H0[2], color="red", s=60, label="H0")

    # Osie główne bezwładności
    axis_len = 1.15 * Hmag

    ax.plot([0, axis_len], [0, 0], [0, 0], "--", color="black", linewidth=0.8)
    ax.plot([0, 0], [0, axis_len], [0, 0], "--", color="black", linewidth=0.8)
    ax.plot([0, 0], [0, 0], [0, axis_len], "--", color="black", linewidth=0.8)

    ax.text(axis_len, 0, 0, "e1")
    ax.text(0, axis_len, 0, "e2")
    ax.text(0, 0, axis_len, "e3")

    ax.set_xlabel("H1")
    ax.set_ylabel("H2")
    ax.set_zlabel("H3")

    if title is None:
        title = f"Constant energy paths on angular momentum sphere\nJ = {J}"

    ax.set_title(title)

    lim = 1.2 * Hmag
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])

    ax.view_init(elev=23, azim=-55)

    if H0 is not None:
        ax.legend()

    plt.show()


# ============================================================
# PRZYKŁAD 1: jak rysunek a z książki
# J1 : J2 : J3 = 4 : 4 : 6
# ciało osiowosymetryczne
# ============================================================

J = [4, 4, 6]
H0 = [0.2, 0.0, 1.0]

plot_energy_paths_on_momentum_sphere(
    J=J, H0=H0, n_levels=25, title="Axisymmetric body: J1:J2:J3 = 4:4:6"
)


# ============================================================
# PRZYKŁAD 2: jak rysunek b z książki
# J1 : J2 : J3 = 3 : 4 : 6
# ciało trójosiowe
# ============================================================

J = [3, 4, 6]
H0 = [0.05, 0.02, 1.0]

plot_energy_paths_on_momentum_sphere(
    J=J, H0=H0, n_levels=35, title="Triaxial body: J1:J2:J3 = 3:4:6"
)


# ============================================================
# PRZYKŁAD 3: start blisko osi pośredniej e2
# To pokaże okolice niestabilnej osi.
# ============================================================

J = [3, 4, 6]
H0 = [0.02, 1.0, 0.02]

plot_energy_paths_on_momentum_sphere(
    J=J, H0=H0, n_levels=35, title="Near intermediate axis e2: unstable region"
)
