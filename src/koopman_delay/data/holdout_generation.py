"""Frozen-distribution holdout generation; no fitting or sample selection."""
import numpy as np
from koopman_delay.systems.three_tank.dynamics import threetanks_recycle_delay
from koopman_delay.systems.lorenz.dynamics import DelayedLorenzGenerator

def simulate_trajectory(split: str, index: int, steps_per_episode: int, seed: int, recycle_delay_tau: float) -> dict:
    rng = np.random.default_rng(int(seed))
    env = threetanks_recycle_delay(recycle_delay_tau=float(recycle_delay_tau), x0_std_dev=0.0)
    center_scale = rng.normal(0.0, 0.025, size=env.x_dim)
    x0 = np.clip(env.xs * (1.0 + center_scale), env.state_low, env.state_high)
    process_noise = env.generate_noise_with_rng(rng, steps_per_episode, noise_type="gaussian", level=0.002)
    env.set_initial(x0, noise=process_noise)
    states = np.empty((steps_per_episode + 1, env.x_dim), dtype=np.float64)
    actions = np.empty((steps_per_episode, env.u_dim), dtype=np.float64)
    states[0] = env.state.copy()
    for step in range(steps_per_episode):
        if step % 2 == 0:
            span = env.action_high - env.action_low
            action = np.clip(env.us + rng.uniform(-0.12, 0.12, size=env.u_dim) * span, env.action_low, env.action_high)
        else:
            action = rng.uniform(env.action_low, env.action_high)
        state, _, terminated, truncated, info = env.step(action, noise=True)
        if terminated or truncated:
            raise RuntimeError(f"unexpected plant termination for {split}_{index}")
        states[step + 1] = state
        actions[step] = action
    return {"states": states, "actions": actions, "initial_state": x0, "info": info}


def generate(system, index, seed):
    if system == "three_tank":
        payload = simulate_trajectory("fresh_holdout", index, 1000, seed, 0.025)
        return payload["states"], payload["actions"]
    if system == "lorenz":
        states = DelayedLorenzGenerator(seed=seed).generate(1000, 500)
        return states, np.empty((len(states)-1, 0), dtype=np.float64)
    raise ValueError(system)
