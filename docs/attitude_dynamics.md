# Attitude Dynamics

## Body Frame

The body frame is attached to the spacecraft.

Initial body axes are defined from the first orbit sample:

$$
\hat{\mathbf{x}}_\mathrm{body}(0)
= \frac{\mathbf{r}_\mathrm{eci}}{\lVert \mathbf{r}_\mathrm{eci} \rVert}
$$

$$
\hat{\mathbf{z}}_\mathrm{body}(0)
= \frac{\mathbf{r}_\mathrm{eci} \times \mathbf{v}_\mathrm{eci}}
       {\lVert \mathbf{r}_\mathrm{eci} \times \mathbf{v}_\mathrm{eci} \rVert}
$$

$$
\hat{\mathbf{y}}_\mathrm{body}(0)
= \hat{\mathbf{z}}_\mathrm{body}(0) \times \hat{\mathbf{x}}_\mathrm{body}(0)
$$

These axes form the initial rotation matrix:

$$
\mathbf{R}_{\mathrm{eci}\leftarrow\mathrm{body}}(0)
=
\begin{bmatrix}
\hat{\mathbf{x}}_\mathrm{body}(0) &
\hat{\mathbf{y}}_\mathrm{body}(0) &
\hat{\mathbf{z}}_\mathrm{body}(0)
\end{bmatrix}
$$

After $t = 0$, the body frame is propagated by rigid-body dynamics.

## Quaternion Convention

Quaternions are scalar-first:

$$
\mathbf{q} = \begin{bmatrix} q_w & q_x & q_y & q_z \end{bmatrix}^\mathsf{T}
$$

The quaternion maps body vectors to ECI through:

$$
\mathbf{v}_\mathrm{eci}
= \mathbf{R}_{\mathrm{eci}\leftarrow\mathrm{body}}(\mathbf{q})\,
  \mathbf{v}_\mathrm{body}
$$

Therefore:

$$
\mathbf{v}_\mathrm{body}
= \mathbf{R}_{\mathrm{eci}\leftarrow\mathrm{body}}(\mathbf{q})^\mathsf{T}\,
  \mathbf{v}_\mathrm{eci}
$$

In code, this rotation is named `rotation_eci_from_body`.

## Quaternion Kinematics

Angular velocity is expressed in body coordinates:

$$
\boldsymbol{\omega}_\mathrm{body}
= \begin{bmatrix} \omega_x & \omega_y & \omega_z \end{bmatrix}^\mathsf{T}
$$

The quaternion equation is:

$$
\dot{\mathbf{q}}
= \frac{1}{2}\,\mathbf{q} \otimes
  \begin{bmatrix} 0 & \boldsymbol{\omega}_\mathrm{body}^\mathsf{T} \end{bmatrix}^\mathsf{T}
$$

where $\otimes$ is quaternion multiplication.

## Euler Rigid-Body Equation

The rotational dynamics are:

$$
\mathbf{I}\dot{\boldsymbol{\omega}}
+ \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})
= \boldsymbol{\tau}
$$

or:

$$
\dot{\boldsymbol{\omega}}
= \mathbf{I}^{-1}
  \left(\boldsymbol{\tau}
  - \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})\right)
$$

Current default:

$$
\boldsymbol{\tau} = \mathbf{0}
$$

so the motion is torque-free.

## Rotation Sanity Checks

For a valid rotation matrix:

$$
\mathbf{R}^\mathsf{T}\mathbf{R} = \mathbf{I}
$$

$$
\det(\mathbf{R}) = 1
$$

The CSV stores:

- all entries of $\mathbf{R}^\mathsf{T}\mathbf{R} - \mathbf{I}$,
- Frobenius norm $\lVert \mathbf{R}^\mathsf{T}\mathbf{R} - \mathbf{I} \rVert_F$,
- determinant $\det(\mathbf{R})$.

These values should stay close to:

$$
\mathbf{R}^\mathsf{T}\mathbf{R} - \mathbf{I} = \mathbf{0}
$$

$$
\det(\mathbf{R}) = 1
$$
