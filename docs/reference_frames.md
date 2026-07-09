# Reference Frames

## ECI

ECI is the inertial frame used by the orbit propagation.

The pipeline assumes that the vectors from `poliastro` can be treated as
GCRS-like ECI vectors for the current engineering model.

## ECEF

ECEF is the Earth-fixed frame.

The position transformation is:

$$
\mathbf{r}_\mathrm{ecef}(t)
= \mathbf{R}_{\mathrm{ecef}\leftarrow\mathrm{eci}}(t)\,
  \mathbf{r}_\mathrm{eci}(t)
$$

In code this is performed with Astropy frame transformations.

## Geodetic Coordinates

ECEF position is converted to WGS84 geodetic coordinates:

$$
\mathbf{r}_\mathrm{ecef}
\rightarrow
(\phi,\lambda,h)
$$

where $\phi$ is latitude, $\lambda$ is longitude and $h$ is altitude.

The project stores:

- `lat_deg`, `lon_deg`,
- `lat_rad`, `lon_rad`,
- `alt_m`, `alt_km`.

## NED Basis

At geodetic latitude $\phi$ and longitude $\lambda$, the local NED basis in ECEF
coordinates is:

$$
\hat{\mathbf{n}}
=
\begin{bmatrix}
-\sin\phi\cos\lambda \\
-\sin\phi\sin\lambda \\
\cos\phi
\end{bmatrix}
$$

$$
\hat{\mathbf{e}}
=
\begin{bmatrix}
-\sin\lambda \\
\cos\lambda \\
0
\end{bmatrix}
$$

$$
\hat{\mathbf{d}}
=
\begin{bmatrix}
-\cos\phi\cos\lambda \\
-\cos\phi\sin\lambda \\
-\sin\phi
\end{bmatrix}
$$

For a local vector

$$
\mathbf{b}_\mathrm{ned}
=
\begin{bmatrix}
B_N & B_E & B_D
\end{bmatrix}^\mathsf{T},
$$

the corresponding ECEF vector is:

$$
\mathbf{b}_\mathrm{ecef}
= B_N\hat{\mathbf{n}} + B_E\hat{\mathbf{e}} + B_D\hat{\mathbf{d}}
$$

## Vector Transformation Back to ECI

Free vectors are transformed back to ECI with:

$$
\mathbf{b}_\mathrm{eci}
= \mathbf{R}_{\mathrm{eci}\leftarrow\mathrm{ecef}}(t)\,
  \mathbf{b}_\mathrm{ecef}
$$

The current code delegates this vector rotation to `pymap3d`/Astropy. This keeps
the frame convention explicit and aligned with the position transformation.
