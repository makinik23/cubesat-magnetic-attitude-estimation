# Mathematical Notes

These notes summarize the mechanics used in the simulation.

Mathematical expressions use MathJax/KaTeX-style Markdown:

- inline math: `$x = y$`,
- display math:

```text
$$
x = y
$$
```

Read in this order:

1. [Orbital mechanics](orbital_mechanics.md)
2. [Reference frames](reference_frames.md)
3. [Geomagnetic field](geomagnetic_field.md)
4. [Attitude dynamics](attitude_dynamics.md)
5. [Magnetometer model](magnetometer_model.md)
6. [Numerical methods](numerical_methods.md)

Main conventions:

- Vectors are column vectors.
- Rotation matrices are named `R_target_from_source`.
- Quaternions are scalar-first: $\mathbf{q} = [q_w, q_x, q_y, q_z]^\mathsf{T}$.
- SI units are used internally unless a column name says otherwise.
- Magnetic field is stored in tesla for vectors and nT for IGRF NED components.
- Magnetometer output is the body-frame field plus configured bias and
  Gaussian noise.
