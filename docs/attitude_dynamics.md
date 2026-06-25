# Attitude Dynamics

## Body Frame

The body frame is attached to the spacecraft.

Initial body axes are defined from the first orbit sample:

```text
x_body(0) = r_eci / ||r_eci||
z_body(0) = (r_eci x v_eci) / ||r_eci x v_eci||
y_body(0) = z_body(0) x x_body(0)
```

These axes form the initial rotation matrix:

```text
R_eci_from_body(0) = [x_body(0) y_body(0) z_body(0)]
```

After `t = 0`, the body frame is propagated by rigid-body dynamics.

## Quaternion Convention

Quaternions are scalar-first:

```text
q = [w, x, y, z]
```

The quaternion maps body vectors to ECI through:

```text
v_eci = R_eci_from_body(q) * v_body
```

Therefore:

```text
v_body = R_eci_from_body(q)^T * v_eci
```

## Quaternion Kinematics

Angular velocity is expressed in body coordinates:

```text
omega_body = [omega_x, omega_y, omega_z]
```

The quaternion equation is:

```text
q_dot = 1/2 * q (*) [0, omega_body]
```

where `(*)` is quaternion multiplication.

## Euler Rigid-Body Equation

The rotational dynamics are:

```text
I * omega_dot + omega x (I * omega) = tau
```

or:

```text
omega_dot = I^-1 * (tau - omega x (I * omega))
```

Current default:

```text
tau = 0
```

so the motion is torque-free.

## Rotation Sanity Checks

For a valid rotation matrix:

```text
R^T R = I
det(R) = 1
```

The CSV stores:

- all entries of `R^T R - I`,
- Frobenius norm `||R^T R - I||_F`,
- determinant `det(R)`.

These values should stay close to:

```text
R^T R - I = 0
det(R) = 1
```
