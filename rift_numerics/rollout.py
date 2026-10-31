"""Frozen float64 Yoshida4 integration for two-dimensional states.

No acceleration law, force parameters, source catalogue, hidden object or
experiment data is provided here. The caller supplies acceleration(t, states)
and is responsible for its physical model, including background propagation.
Time advances in every drift substep. Only requested public times must align
with the fixed time grid; the internal Yoshida substeps are not rounded.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

DT = 0.005
NUMERICS_VERSION = "1.1"
IMPLEMENTATION_VERSION = "rift-numerics-1.0.0"
ALIGNMENT_TOLERANCE_TICKS = 1e-9
_W = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_C = (_W / 2, (1 - _W) / 2, (1 - _W) / 2, _W / 2)
YOSHIDA_D = (_W, 1 - 2 * _W, _W)


def _scalar(value, name):
    if isinstance(value, (bool, str, bytes)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _tick(value, name, step):
    number = _scalar(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative.")
    ratio = number / step
    if not np.isfinite(ratio):
        raise ValueError(f"{name} is out of range.")
    result = round(ratio)
    if abs(ratio - result) > ALIGNMENT_TOLERANCE_TICKS:
        raise ValueError(f"{name} must lie on the {step} time grid.")
    return result


def _state(value, name):
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (N, 2).") from exc
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape (N, 2).")
    return array


def sample_indices(relative_times):
    """Validate strictly increasing public offsets and return integer ticks."""
    if isinstance(relative_times, (str, bytes)):
        raise ValueError("relative_times must be a nonempty sequence.")
    try:
        values = list(relative_times)
    except TypeError as exc:
        raise ValueError("relative_times must be a nonempty sequence.") from exc
    if not values:
        raise ValueError("relative_times must be nonempty.")
    result = np.array([_tick(t, "relative_time", DT) for t in values], dtype=np.int64)
    if np.any(np.diff(result) <= 0):
        raise ValueError("relative_times must be strictly increasing.")
    return result


def _integrate(acceleration: Callable, positions, velocities, t_launch, duration, dt):
    if not callable(acceleration):
        raise ValueError("acceleration must be callable.")
    pos = _state(positions, "positions")
    vel = _state(velocities, "velocities")
    if pos.shape != vel.shape:
        raise ValueError("Positions and velocities must have identical shape.")
    step_size = _scalar(dt, "dt")
    if step_size <= 0:
        raise ValueError("dt must be positive.")
    steps = _tick(duration, "duration", step_size)
    start = _scalar(t_launch, "t_launch")
    if start < 0:
        raise ValueError("t_launch must be nonnegative.")
    positions_out = np.empty((steps + 1, *pos.shape), dtype=np.float64)
    velocities_out = np.empty_like(positions_out)
    positions_out[0], velocities_out[0] = pos, vel
    for step in range(steps):
        instant = start + step * step_size
        for c, d in zip(YOSHIDA_C[:-1], YOSHIDA_D):
            pos = pos + c * step_size * vel
            instant += c * step_size
            result = np.asarray(acceleration(instant, pos), dtype=np.float64)
            if result.shape != pos.shape:
                raise ValueError("acceleration must return the same (N, 2) shape as positions.")
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("Acceleration returned a nonfinite value.")
            vel = vel + d * step_size * result
        pos = pos + YOSHIDA_C[-1] * step_size * vel
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            raise FloatingPointError("Integration produced a nonfinite state.")
        positions_out[step + 1], velocities_out[step + 1] = pos, vel
    return {"times": start + np.arange(steps + 1, dtype=np.float64) * step_size,
            "positions": positions_out, "velocities": velocities_out}


def rollout(acceleration: Callable, positions, velocities, *, t_launch=0., duration=1.):
    """Integrate with the frozen public step and retain all grid states.

    Inputs are initial states at absolute physical t_launch. Integrating a
    background before a later launch is the caller's responsibility. Callback
    inputs are the current substep states, including any background objects
    placed into the initial state array by the caller.
    """
    _tick(t_launch, "t_launch", DT)
    return _integrate(acceleration, positions, velocities, t_launch, duration, DT)


def diagnostic_rollout(acceleration: Callable, positions, velocities, *,
                       t_launch=0., duration=1., dt=DT):
    """Organizer numerical-convergence entry point; never a team-selected step.

    It shares the exact same integration function as rollout. Public request
    validation and trusted formal prediction still use the fixed-step entry.
    """
    return _integrate(acceleration, positions, velocities, t_launch, duration, dt)
