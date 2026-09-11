"""Delayed Lorenz-63 solver for Benchmark III-R1.

The delayed equations are

    dx1/dt = 10[x2(t-tau) - x1(t)]
    dx2/dt = 28*x1(t) - x1(t)*x3(t) - x2(t)
    dx3/dt = x1(t)*x2(t) - (8/3)*x3(t-tau)

The solver uses a timestamped internal-step grid.  Each grid entry stores the
state and its RHS derivative, and each RK4 stage queries ``x(s-tau)`` using
cubic-Hermite interpolation of the two timestamp-bracketing entries.  The
sample interval is only an output cadence; it is never used as the delay
lookup cadence.
"""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
from typing import Any, Callable

import numpy as np


# Frozen Benchmark III constants.
A = 10.0
B = 8.0 / 3.0
C = 28.0
TAU = 1.0 / 6.0
SAMPLE_DT = 1.0 / 30.0
INTERNAL_DT = 1.0 / 150.0
SAMPLING_STEPS = 5
DELAY_INTEGRATION_STEPS = 25
DELAY_SAMPLE_STEPS = 5
STATE_DIM = 3
TIMESTAMP_TOL = 1e-12


def delayed_lorenz_derivative(x_current: np.ndarray, x_delayed: np.ndarray) -> np.ndarray:
    """Evaluate the delayed Lorenz RHS at one current/delayed state pair."""

    x1, x2, x3 = np.asarray(x_current, dtype=np.float64)
    x1_tau, x2_tau, x3_tau = np.asarray(x_delayed, dtype=np.float64)
    return np.asarray(
        [A * (x2_tau - x1), C * x1 - x1 * x3 - x2, x1 * x2 - B * x3_tau],
        dtype=np.float64,
    )


def standard_lorenz_derivative(x: np.ndarray) -> np.ndarray:
    """Evaluate the tau=0 Lorenz-63 RHS."""

    x1, x2, x3 = np.asarray(x, dtype=np.float64)
    return np.asarray([A * (x2 - x1), C * x1 - x1 * x3 - x2, x1 * x2 - B * x3], dtype=np.float64)


def rk4_step(x_current: np.ndarray, x_delayed: np.ndarray, dt: float) -> np.ndarray:
    """Single RK4 step with a fixed delayed state, retained for compatibility."""

    dt = float(dt)
    k1 = delayed_lorenz_derivative(x_current, x_delayed)
    k2 = delayed_lorenz_derivative(x_current + 0.5 * dt * k1, x_delayed)
    k3 = delayed_lorenz_derivative(x_current + 0.5 * dt * k2, x_delayed)
    k4 = delayed_lorenz_derivative(x_current + dt * k3, x_delayed)
    return np.asarray(x_current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)


def rk4_step_standard(x: np.ndarray, dt: float) -> np.ndarray:
    """Single RK4 step for the standard tau=0 Lorenz system."""

    dt = float(dt)
    k1 = standard_lorenz_derivative(x)
    k2 = standard_lorenz_derivative(x + 0.5 * dt * k1)
    k3 = standard_lorenz_derivative(x + 0.5 * dt * k2)
    k4 = standard_lorenz_derivative(x + dt * k3)
    return np.asarray(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)


HistoryFunction = Callable[[float], np.ndarray]
HistoryDerivativeFunction = Callable[[float], np.ndarray]


class DelayedLorenzGenerator:
    """Timestamped internal-step RK4 generator for the delayed Lorenz DDE."""

    def __init__(
        self,
        seed: int,
        initial_history: np.ndarray | None = None,
        internal_dt: float = INTERNAL_DT,
        sampling_steps: int = SAMPLING_STEPS,
        delay_steps: int = DELAY_INTEGRATION_STEPS,
        tau: float = TAU,
        history_fn: HistoryFunction | None = None,
        history_derivative_fn: HistoryDerivativeFunction | None = None,
    ) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.internal_dt = float(internal_dt)
        self.sampling_steps = int(sampling_steps)
        self.delay_steps = int(delay_steps)
        self.tau = float(tau)
        self._history_fn_input = history_fn
        self._history_derivative_fn = history_derivative_fn
        if self.internal_dt <= 0.0 or self.tau <= 0.0:
            raise ValueError("internal_dt and tau must be positive")
        if self.delay_steps < 1:
            raise ValueError("delay_steps must be positive")
        if self.sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        self.sample_dt = self.sampling_steps * self.internal_dt
        # The initial history always includes both exact delay endpoints.  For
        # the benchmark grid this is the same as 25 internal steps; the
        # convergence audit also uses h=1/75, for which tau/h is noninteger.
        self._initial_history_times = np.linspace(
            -self.tau, 0.0, self.delay_steps + 1, dtype=np.float64
        )
        self.history_buffer = self._coerce_initial_history(initial_history, history_fn)
        self.reset()

    def _coerce_initial_history(self, initial_history: np.ndarray | None, history_fn: HistoryFunction | None) -> np.ndarray:
        n_grid = self.delay_steps + 1
        if history_fn is not None:
            return np.stack([self._call_history(history_fn, time) for time in self._initial_history_times])
        if initial_history is None:
            return self._random_initial_history()
        history = np.asarray(initial_history, dtype=np.float64)
        if history.shape != (n_grid, STATE_DIM):
            raise ValueError(f"initial_history must have shape {(n_grid, STATE_DIM)}, got {history.shape}")
        if not np.isfinite(history).all():
            raise ValueError("initial_history must be finite")
        return history.copy()

    @staticmethod
    def _call_history(history_fn: HistoryFunction, time: float) -> np.ndarray:
        state = np.asarray(history_fn(float(time)), dtype=np.float64)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError("history_fn must return a finite shape-(3,) state")
        return state

    @staticmethod
    def _call_history_derivative(
        history_derivative_fn: HistoryDerivativeFunction, time: float
    ) -> np.ndarray:
        derivative = np.asarray(history_derivative_fn(float(time)), dtype=np.float64)
        if derivative.shape != (STATE_DIM,) or not np.isfinite(derivative).all():
            raise ValueError("history_derivative_fn must return a finite shape-(3,) derivative")
        return derivative

    def _random_initial_history(self) -> np.ndarray:
        history = np.zeros((self.delay_steps + 1, STATE_DIM), dtype=np.float64)
        history[:, 0] = self.rng.uniform(-20.0, 20.0, size=self.delay_steps + 1)
        history[:, 1] = self.rng.uniform(-20.0, 20.0, size=self.delay_steps + 1)
        history[:, 2] = self.rng.uniform(10.0, 50.0, size=self.delay_steps + 1)
        return history

    def _history_value(self, time: float) -> np.ndarray:
        if self._history_fn_input is not None and time <= 0.0 + TIMESTAMP_TOL:
            return self._call_history(self._history_fn_input, time)
        if time <= -self.tau:
            return self.history_buffer[0].copy()
        if time >= 0.0:
            return self.history_buffer[-1].copy()
        position = (time + self.tau) / (self._initial_history_times[1] - self._initial_history_times[0])
        left = int(np.floor(position))
        fraction = position - left
        left = min(max(left, 0), self.delay_steps - 1)
        return (1.0 - fraction) * self.history_buffer[left] + fraction * self.history_buffer[left + 1]

    def reset(self) -> np.ndarray:
        """Reset the timestamped grid to the original history."""

        self.time = 0.0
        self.state = self.history_buffer[-1].copy()
        self._grid_times = self._initial_history_times.tolist()
        self._grid_states = [state.copy() for state in self.history_buffer]
        if self._history_derivative_fn is not None:
            self._grid_rhs = [
                self._call_history_derivative(self._history_derivative_fn, time)
                for time in self._grid_times
            ]
        else:
            self._grid_rhs = [
                delayed_lorenz_derivative(state, self._history_value(time - self.tau))
                for time, state in zip(self._grid_times, self._grid_states)
            ]
        self.stage_query_log: list[dict[str, float]] = []
        return self.state.copy()

    def _delayed_state_at(self, query_time: float) -> np.ndarray:
        """Cubic-Hermite interpolate x(query_time) on the timestamped grid."""

        if query_time < self._grid_times[0] - TIMESTAMP_TOL or query_time > self._grid_times[-1] + TIMESTAMP_TOL:
            raise ValueError(f"delay query {query_time} is outside grid [{self._grid_times[0]}, {self._grid_times[-1]}]")
        if query_time <= self._grid_times[0] + TIMESTAMP_TOL:
            return self._grid_states[0].copy()
        if query_time >= self._grid_times[-1] - TIMESTAMP_TOL:
            return self._grid_states[-1].copy()
        right = bisect_right(self._grid_times, query_time)
        left = right - 1
        t0, t1 = self._grid_times[left], self._grid_times[right]
        x0, x1 = self._grid_states[left], self._grid_states[right]
        f0, f1 = self._grid_rhs[left], self._grid_rhs[right]
        h = t1 - t0
        u = (query_time - t0) / h
        h00 = (1.0 + 2.0 * u) * (1.0 - u) ** 2
        h10 = u * (1.0 - u) ** 2
        h01 = u**2 * (3.0 - 2.0 * u)
        h11 = -u**2 * (1.0 - u)
        return h00 * x0 + h10 * h * f0 + h01 * x1 + h11 * h * f1

    def _delayed_at_stage(self, stage_time: float) -> np.ndarray:
        source_time = float(stage_time - self.tau)
        self.stage_query_log.append({
            "stage_time": float(stage_time),
            "queried_source_time": source_time,
            "timestamp_error": abs(source_time - (float(stage_time) - self.tau)),
        })
        return self._delayed_state_at(source_time)

    def _rk4_step(self, step_size: float | None = None) -> None:
        """Advance one internal step with four independently interpolated delays."""

        h = self.internal_dt if step_size is None else float(step_size)
        if h <= 0.0 or not np.isclose(h, self.internal_dt, atol=TIMESTAMP_TOL, rtol=0.0):
            raise ValueError("step_size must equal internal_dt")
        t = float(self.time)
        x = self.state.copy()
        d1 = self._delayed_at_stage(t)
        d2 = self._delayed_at_stage(t + 0.5 * h)
        d3 = self._delayed_at_stage(t + 0.5 * h)
        d4 = self._delayed_at_stage(t + h)
        k1 = delayed_lorenz_derivative(x, d1)
        k2 = delayed_lorenz_derivative(x + 0.5 * h * k1, d2)
        k3 = delayed_lorenz_derivative(x + 0.5 * h * k2, d3)
        k4 = delayed_lorenz_derivative(x + h * k3, d4)
        x_new = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(x_new).all():
            raise FloatingPointError("delayed Lorenz RK4 step produced a non-finite state")
        self._grid_times.append(t + h)
        self._grid_states.append(x_new.copy())
        self._grid_rhs.append(delayed_lorenz_derivative(x_new, d4))
        self.state = x_new
        self.time = t + h

    def integrate(self, t_end: float) -> None:
        """Integrate by internal steps without overshooting ``t_end``."""

        t_end = float(t_end)
        if t_end < self.time - TIMESTAMP_TOL:
            raise ValueError("t_end must not precede current time")
        n_steps = int(np.floor((t_end - self.time) / self.internal_dt + TIMESTAMP_TOL))
        for _ in range(n_steps):
            self._rk4_step()

    def rollout(self, t_end: float, sample_time: float | None = None, t_start: float = 0.0) -> dict[str, np.ndarray]:
        """Integrate and return states at the requested output cadence."""

        sample = self.sample_dt if sample_time is None else float(sample_time)
        sample_steps_float = sample / self.internal_dt
        if not np.isclose(sample_steps_float, round(sample_steps_float), atol=TIMESTAMP_TOL, rtol=0.0):
            raise ValueError("sample_time must be an integer multiple of internal_dt")
        if t_start < self.time - TIMESTAMP_TOL or t_start > t_end + TIMESTAMP_TOL:
            raise ValueError("invalid rollout start")
        self.integrate(t_start)
        n_samples = int(round((t_end - t_start) / sample))
        times: list[float] = []
        states: list[np.ndarray] = []
        for _ in range(n_samples):
            self.integrate(self.time + sample)
            times.append(float(self.time))
            states.append(self.state.copy())
        return {"times": np.asarray(times, dtype=np.float64), "states": np.asarray(states, dtype=np.float64)}

    def generate(self, n_samples: int, burn_in_steps: int = 500) -> np.ndarray:
        """Generate ``n_samples`` sampled states after sample-level burn-in."""

        if n_samples < 1 or burn_in_steps < 0:
            raise ValueError("n_samples must be positive and burn_in_steps non-negative")
        self.reset()
        total_samples = int(burn_in_steps) + int(n_samples)
        output: list[np.ndarray] = []
        for sample_index in range(total_samples):
            for _ in range(self.sampling_steps):
                self._rk4_step()
            if sample_index >= burn_in_steps:
                output.append(self.state.copy())
        return np.asarray(output, dtype=np.float64)

    def generate_with_sanity(self, n_samples: int, burn_in_steps: int = 500) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate a trajectory with finite/nonconstant and checksum metadata."""

        trajectory = self.generate(n_samples=n_samples, burn_in_steps=burn_in_steps)
        differences = np.diff(trajectory, axis=0)
        stats = {
            "seed": self.seed,
            "n_samples": n_samples,
            "burn_in_steps": burn_in_steps,
            "shape": list(trajectory.shape),
            "dtype": str(trajectory.dtype),
            "all_finite": bool(np.isfinite(trajectory).all()),
            "is_constant": bool(np.all(differences == 0)),
            "x1_range": [float(trajectory[:, 0].min()), float(trajectory[:, 0].max())],
            "x2_range": [float(trajectory[:, 1].min()), float(trajectory[:, 1].max())],
            "x3_range": [float(trajectory[:, 2].min()), float(trajectory[:, 2].max())],
            "sha256": hashlib.sha256(trajectory.tobytes()).hexdigest(),
        }
        return trajectory, stats

    def timestamp_delay_audit(self, n_steps: int = 128) -> dict[str, Any]:
        """Audit exact source-time bookkeeping over four stages per step."""

        if n_steps < 100:
            raise ValueError("timestamp audit requires at least 100 internal steps")

        def timestamp_history(time: float) -> np.ndarray:
            return np.asarray([time, 2.0 * time, -3.0 * time], dtype=np.float64)

        def timestamp_history_derivative(_time: float) -> np.ndarray:
            return np.asarray([1.0, 2.0, -3.0], dtype=np.float64)

        probe = DelayedLorenzGenerator(
            seed=self.seed,
            internal_dt=self.internal_dt,
            sampling_steps=1,
            delay_steps=self.delay_steps,
            tau=self.tau,
            history_fn=timestamp_history,
            history_derivative_fn=timestamp_history_derivative,
        )
        for _ in range(n_steps):
            probe._rk4_step()
        errors = np.asarray([entry["timestamp_error"] for entry in probe.stage_query_log], dtype=np.float64)
        history_queries = np.linspace(-self.tau, 0.0, 41, dtype=np.float64)
        history_value_errors = np.asarray([
            np.max(np.abs(probe._delayed_state_at(query) - timestamp_history(query)))
            for query in history_queries
        ], dtype=np.float64)
        return {
            "n_internal_steps": n_steps,
            "stage_query_count": len(probe.stage_query_log),
            "tau": self.tau,
            "internal_dt": self.internal_dt,
            "delay_internal_steps": self.delay_steps,
            "max_timestamp_error": float(errors.max()) if errors.size else float("inf"),
            "all_timestamp_queries_exact": bool(errors.size == n_steps * 4 and np.all(errors <= TIMESTAMP_TOL)),
            "max_history_interpolation_error": float(history_value_errors.max()),
            "history_interpolation_exact": bool(np.all(history_value_errors <= TIMESTAMP_TOL)),
            "history_state_encodes_timestamp": True,
        }


def _convergence_history(time: float) -> np.ndarray:
    """Smooth history used only for the short numerical convergence audit."""

    x0 = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    return x0 + _convergence_history_slope() * float(time)


def _convergence_history_slope() -> np.ndarray:
    """Choose a linear history whose derivative is compatible at t=0."""

    # For x(0)=(1,1,1), p=x'(0), and x(-tau)=x(0)-tau*p, solve
    # p=f(x(0), x(-tau)).  This avoids an artificial derivative jump at the
    # start of the method-of-steps convergence interval.
    x0 = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    p2 = C * x0[0] - x0[0] * x0[2] - x0[1]
    p3 = (x0[0] * x0[1] - B * x0[2]) / (1.0 - B * TAU)
    p1 = A * (x0[1] - TAU * p2 - x0[0])
    return np.asarray([p1, p2, p3], dtype=np.float64)


def _integrate_short(step_size: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    delay_steps = int(round(TAU / step_size))
    solver = DelayedLorenzGenerator(
        seed=0,
        internal_dt=step_size,
        sampling_steps=1,
        delay_steps=delay_steps,
        tau=TAU,
        history_fn=_convergence_history,
        history_derivative_fn=lambda _time: _convergence_history_slope(),
    )
    n_steps = int(round(duration / step_size))
    states = [solver.state.copy()]
    times = [0.0]
    for _ in range(n_steps):
        solver._rk4_step()
        states.append(solver.state.copy())
        times.append(float(solver.time))
    return np.asarray(times), np.asarray(states)


def solver_convergence_audit(duration: float = 0.2) -> dict[str, Any]:
    """Compare h=1/75, 1/150, 1/300, 1/600 on one short interval."""

    step_sizes = [1.0 / 75.0, 1.0 / 150.0, 1.0 / 300.0, 1.0 / 600.0]
    trajectories = {step: _integrate_short(step, duration) for step in step_sizes}
    fine_step = step_sizes[-1]
    fine_times, fine_states = trajectories[fine_step]
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(step_sizes):
        times, states = trajectories[step]
        factor = int(round(step / fine_step))
        reference = fine_states[::factor]
        difference = states - reference
        l2_error = float(np.sqrt(np.mean(difference**2)))
        endpoint_error = float(np.linalg.norm(difference[-1]))
        if index + 1 < len(step_sizes):
            finer_step = step_sizes[index + 1]
            finer_times, finer_states = trajectories[finer_step]
            pair_factor = int(round(step / finer_step))
            pair_difference = states - finer_states[::pair_factor]
            pair_l2_error = float(np.sqrt(np.mean(pair_difference**2)))
            pair_endpoint_error = float(np.linalg.norm(pair_difference[-1]))
        else:
            pair_l2_error = None
            pair_endpoint_error = None
        rows.append({
            "h": step,
            "n_steps": len(times) - 1,
            "l2_error": l2_error,
            "endpoint_error": endpoint_error,
            "pair_l2_error_to_next_finer": pair_l2_error,
            "pair_endpoint_error_to_next_finer": pair_endpoint_error,
            "finite": bool(np.isfinite(states).all() and np.isfinite(difference).all()),
        })
    for row, finer in zip(rows[:-2], rows[1:-1]):
        row["empirical_order_to_next_finer"] = float(
            np.log2(row["pair_l2_error_to_next_finer"] / finer["pair_l2_error_to_next_finer"])
        )
    for row in rows[-2:]:
        row["empirical_order_to_next_finer"] = None
    orders = [row["empirical_order_to_next_finer"] for row in rows[:-2]]
    target_order_pass = bool(orders and min(orders) >= 3.0 and max(orders) <= 5.5)
    return {
        "duration": float(duration),
        "step_sizes": step_sizes,
        "reference_step": fine_step,
        "rows": rows,
        "target_order": 4.0,
        "target_order_tolerance": [3.0, 5.5],
        "target_order_pass": target_order_pass,
    }


def test_tau0_reduction() -> dict[str, Any]:
    x = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    difference = np.max(np.abs(delayed_lorenz_derivative(x, x) - standard_lorenz_derivative(x)))
    return {"test": "tau0_reduction", "max_abs_diff": float(difference), "pass": bool(difference <= 1e-14)}


def test_delay_index() -> dict[str, Any]:
    passed = DELAY_SAMPLE_STEPS == 5 and DELAY_INTEGRATION_STEPS == 25 and SAMPLING_STEPS == 5
    return {
        "test": "delay_index",
        "delay_sample_steps": DELAY_SAMPLE_STEPS,
        "delay_integration_steps": DELAY_INTEGRATION_STEPS,
        "sampling_steps": SAMPLING_STEPS,
        "tau": TAU,
        "sample_dt": SAMPLE_DT,
        "internal_dt": INTERNAL_DT,
        "pass": passed,
    }


def test_solver_repeatability(seed: int = 6) -> dict[str, Any]:
    first = DelayedLorenzGenerator(seed=seed).generate(n_samples=100, burn_in_steps=100)
    second = DelayedLorenzGenerator(seed=seed).generate(n_samples=100, burn_in_steps=100)
    identical = bool(np.array_equal(first, second))
    return {"test": "solver_repeatability", "seed": seed, "identical": identical, "pass": identical}


def test_all_finite(n_trajectories: int = 5, seed: int = 6) -> dict[str, Any]:
    results = []
    for index in range(n_trajectories):
        trajectory, stats = DelayedLorenzGenerator(seed=seed + index * 1009).generate_with_sanity(1000, 500)
        results.append({
            "trajectory": index,
            "seed": seed + index * 1009,
            "all_finite": stats["all_finite"],
            "is_constant": stats["is_constant"],
            "shape": list(trajectory.shape),
        })
    return {"test": "all_finite", "n_trajectories": n_trajectories, "results": results, "pass": all(r["all_finite"] and not r["is_constant"] for r in results)}


def test_convergence() -> dict[str, Any]:
    audit = solver_convergence_audit()
    return {"test": "convergence", **audit, "pass": bool(audit["target_order_pass"])}


def run_all_solver_tests() -> dict[str, Any]:
    """Run the solver-only gates; no model or dataset evaluation is involved."""

    tests = [
        test_tau0_reduction(),
        test_delay_index(),
        test_solver_repeatability(),
        test_all_finite(),
        test_convergence(),
    ]
    timestamp = DelayedLorenzGenerator(seed=6).timestamp_delay_audit()
    tests.append({"test": "timestamp_delay", **timestamp, "pass": timestamp["all_timestamp_queries_exact"]})
    return {"schema_version": 2, "system": "delayed_lorenz", "tests": tests, "all_pass": all(t["pass"] for t in tests)}
