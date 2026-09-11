"""Full causal delayed-plant reconstruction for Figure 5 interventions."""
import copy
import numpy as np
from koopman_delay.systems.three_tank.dynamics import threetanks_recycle_delay

ORIGIN = 459
HORIZON = 100
BASE_TOLERANCE = 1e-10
PROFILE_NAMES = ("base", "q1_plus", "q1_minus", "q2_plus", "q2_minus", "q3_plus", "q3_minus")


def reconstruct(seed, states, actions):
    """Replay every past action; preserve the original internal delay buffer."""
    rng = np.random.default_rng(int(seed))
    env = threetanks_recycle_delay(recycle_delay_tau=.025, x0_std_dev=0.0)
    center = rng.normal(0, .025, size=env.x_dim)
    x0 = np.clip(env.xs * (1 + center), env.state_low, env.state_high)
    noise = env.generate_noise_with_rng(rng, 1000, noise_type="gaussian", level=.002)
    env.set_initial(x0, noise=noise)
    past_error = float(np.max(np.abs(env.state-states[0])))
    for step in range(ORIGIN):
        value = env.step(actions[step], noise=True)[0]
        past_error = max(past_error, float(np.max(np.abs(value-states[step+1]))))
    if past_error > BASE_TOLERANCE:
        raise RuntimeError(f"Full past replay mismatch: {past_error}")
    return env, noise, past_error


def profiles(base, std, low, high):
    requested = np.repeat(np.asarray(base)[None], 7, axis=0)
    for channel in range(3):
        requested[1+2*channel,:,channel] += std[channel]
        requested[2+2*channel,:,channel] -= std[channel]
    applied = np.clip(requested, low, high)
    clipped = np.count_nonzero(requested != applied, axis=1)
    return requested, applied, clipped


def branch_truth(env, actions, future_noise):
    """Change only future forcing; past state, noise and delay buffer stay intact."""
    branch = copy.deepcopy(env)
    start = ORIGIN * branch.sampling_steps
    branch.noise[start:start+HORIZON*branch.sampling_steps] = future_noise
    return np.asarray([branch.step(action, noise=True)[0] for action in actions])
