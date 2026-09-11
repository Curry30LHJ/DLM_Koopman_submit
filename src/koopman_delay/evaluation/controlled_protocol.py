import hashlib
import numpy as np

NOISE_BASE_A = 4_100_000_000
NOISE_BASE_B = 4_200_000_000


def sha_of(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _env():
    from koopman_delay.systems.three_tank.dynamics import threetanks_recycle_delay

    return threetanks_recycle_delay(recycle_delay_tau=0.025, x0_std_dev=0.0)


def gen_noise(rng_seed):
    env = _env()
    rng = np.random.default_rng(int(rng_seed))
    rng.normal(0, 0.025, size=env.x_dim)
    return env.generate_noise_with_rng(rng, 1000, noise_type="gaussian", level=0.002)


def plant_frozen_noise(hist_states, actions_phys, noise_array):
    env = _env()
    sample = np.asarray(hist_states[-6:], float)
    ts = -np.arange(6)[::-1] * 0.005
    its = -np.arange(26)[::-1] * 0.001
    internal = np.stack([np.interp(its, ts, sample[:, d]) for d in range(9)], 1)
    env.recycle_buffer.clear()
    for state in internal:
        env.recycle_buffer.append(env._sanitize_state(state))
    env.state = env._sanitize_state(internal[-1])
    env.noise = noise_array
    env.t = 19
    out = np.empty((100, 9))
    for i, a in enumerate(actions_phys):
        out[i] = env.step(a, noise=True)[0]
    return out


def clip_actions(act_phys):
    env = _env()
    lo, hi = np.asarray(env.action_low, float), np.asarray(env.action_high, float)
    clipped = np.clip(act_phys, lo, hi)
    # exact comparison (rtol=0, atol=0) per boss §9 -- heat-duty values ~1e6
    changed = act_phys != clipped
    n = int(np.count_nonzero(changed))
    return clipped, n
