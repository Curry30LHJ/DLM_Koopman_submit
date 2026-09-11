"""
Three-tank reactor/separator process with recycle-state transport delay.
"""

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    class _Env:
        pass

    class _Box:
        def __init__(self, low, high):
            self.low = np.asarray(low)
            self.high = np.asarray(high)
            self.shape = self.low.shape

        def sample(self):
            return np.random.uniform(self.low, self.high)

    gym = SimpleNamespace(Env=_Env)
    spaces = SimpleNamespace(Box=_Box)


class threetanks_recycle_delay(gym.Env):
    def __init__(
        self,
        x0_std_dev=0.2,
        us=np.array([2.87e6, 1.0e6, 2.87e6]),
        xs=np.array([
            1.920462734802070026e-01,
            6.753561181469071029e-01,
            4.768612896432074422e+02,
            2.116736585616537936e-01,
            6.560585575539794601e-01,
            4.695016333820219074e+02,
            7.212661966828316784e-02,
            6.895290277477214014e-01,
            4.715140681309391084e+02,
        ]),
        recycle_delay_tau=0.025,
    ):
        self.t = 0
        self.action_sample_period = 80
        self.sampling_period = 0.005
        self.h = 0.001
        self.sampling_steps = int(self.sampling_period / self.h)
        self.delay = 0
        self.observed_dims = [2, 5, 8]
        self.x0_std_dev = x0_std_dev

        self.x_dim = 9
        self.y_dim = 3
        self.u_dim = 3
        self.p_dim = 0

        self.s2hr = 3600
        self.MW = 250e-3
        self.sum_c = 2e3
        self.T10 = 300
        self.T20 = 300
        self.F10 = 5.04
        self.F20 = 5.04
        self.Fr = 50.4
        self.Fp = 0.504
        self.V1 = 1.0
        self.V2 = 0.5
        self.V3 = 1.0
        self.E1 = 5e4
        self.E2 = 6e4
        self.k1 = 2.77e3 * self.s2hr
        self.k2 = 2.6e3 * self.s2hr
        self.dH1 = -6e1 / self.MW
        self.dH2 = -7e1 / self.MW
        self.aA = 3.5
        self.aB = 1.0
        self.aC = 0.5
        self.Cp = 4.2
        self.R = 8.314
        self.rho = 1000.0
        self.xA10 = 1.0
        self.xB10 = 0.0
        self.xA20 = 1.0
        self.xB20 = 0.0
        self.Hvap1 = -3.53e1 * self.sum_c
        self.Hvap2 = -1.57e1 * self.sum_c
        self.Hvap3 = -4.068e1 * self.sum_c

        self.kw = np.array([1, 1, 5, 1, 1, 5, 1, 1, 5])
        self.bw = np.array([1, 1, 5, 1, 1, 5, 1, 1, 5])
        self.aw = self.kw
        self.noise = None

        self.xs = np.array([
            0.1763,
            0.6731,
            480.3165,
            0.1965,
            0.6536,
            472.7863,
            0.0651,
            0.6703,
            474.8877,
        ])
        self.ys = self.xs[[2, 5, 8]]
        self.us = np.array(us, dtype=float)

        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.action_low = np.array([1.8e6, 0.2e6, 1.8e6], dtype=np.float32)
        self.action_high = np.array([4e6, 1.68e6, 4e6], dtype=np.float32)
        self.y_low = np.array([0, 0, 0])
        self.y_high = np.array([700, 700, 700])

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)
        self.observation_space = spaces.Box(-high, high)
        self.state_low = np.array([0.0, 0.0, 300.0, 0.0, 0.0, 300.0, 0.0, 0.0, 300.0])
        self.state_high = np.array([1.0, 1.0, 600.0, 1.0, 1.0, 600.0, 1.0, 1.0, 600.0])
        self._eps = 1e-8

        self.recycle_delay_tau = float(recycle_delay_tau)
        self.recycle_delay_steps = int(round(self.recycle_delay_tau / self.h))
        self.recycle_buffer = deque(maxlen=max(self.recycle_delay_steps + 1, 1))

        self.seed()
        self.state_buffer = state_buffer(self.delay)
        self.viewer = None
        self.state = self.xs.copy()
        self.steps_beyond_done = None
        self._reset_recycle_buffer()

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return seed

    def step(self, action, noise=False, observation_noise=False):
        action = np.clip(action, self.action_low, self.action_high)

        x0 = self._sanitize_state(self.state)
        numerical_recovery = False
        clamp_count = 0
        for i in range(self.sampling_steps):
            prev_valid = x0.copy()
            process_noise = (
                np.random.normal(np.zeros_like(self.kw), self.kw)
                if self.noise is None
                else self.noise[self.t * self.sampling_steps + i]
            )
            x_tau = x0 if self.recycle_delay_steps == 0 else self._get_recycle_delayed_state()
            dx = self.derivative(x0, action, x_tau=x_tau)
            if not np.isfinite(dx).all():
                numerical_recovery = True
                dx = np.zeros_like(prev_valid)
            candidate = x0 + dx * self.h
            candidate = candidate + process_noise * self.h if noise else candidate
            candidate_sanitized = self._sanitize_state(candidate, fallback=prev_valid)
            if not np.allclose(candidate, candidate_sanitized, rtol=0.0, atol=0.0):
                clamp_count += 1
            x0 = candidate_sanitized
            self.recycle_buffer.append(x0.copy())

        self.state = x0
        self.t += 1
        self.state_buffer.memorize(self.state.copy())
        self.state_buffer.memorize_u(action.copy())
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "recycle_delay_tau": self.recycle_delay_tau,
            "recycle_delay_steps": self.recycle_delay_steps,
            "measurement_delay": False,
            "delay_type": "recycle_state_delay",
            "numerical_recovery": numerical_recovery,
            "clamp_count": clamp_count,
        }
        return self.state.copy(), reward, terminated, truncated, info

    def step_ob_noise(self, action, noise=False):
        x, reward, terminated, truncated, info = self.step(action, noise=False)
        if noise:
            process_noise = np.random.normal(np.zeros_like(self.kw), self.kw)
            x = x + process_noise * self.h
        return x, reward, terminated, truncated, info

    def reset(self, test=False, seed_=1):
        self.state_buffer.reset()
        self.a_holder = self.action_space.sample()
        self.state = self.xs + np.random.normal(np.zeros_like(self.xs), self.xs * self.x0_std_dev)
        if test:
            np.random.seed(seed_)
            self.state = np.array(
                [0.8508, 0.9805, 0.9786, 0.8906, 0.9791, 0.9772, 0.8449, 0.9697, 0.9791]
            ) * self.xs
        self.state = self._sanitize_state(self.state)
        self.noise = None
        self.t = 0
        self.time = 0
        self._reset_recycle_buffer()
        self.state_buffer.memorize(self.state.copy())
        return self.state.copy(), {}

    def set_initial(self, x0=None, noise=None):
        self.reset()
        self.state = self._sanitize_state(np.asarray(x0, dtype=float)) if x0 is not None else self.state
        self.noise = noise if noise is not None else self.noise
        self.state_buffer.reset()
        self.state_buffer.memorize(self.state.copy())
        self.t = 0
        self.time = 0
        self._reset_recycle_buffer()

    def _reset_recycle_buffer(self):
        self.recycle_buffer.clear()
        for _ in range(max(self.recycle_delay_steps + 1, 1)):
            self.recycle_buffer.append(self.state.copy())

    def _get_recycle_delayed_state(self):
        if not self.recycle_buffer:
            return self.state.copy()
        if len(self.recycle_buffer) < self.recycle_delay_steps + 1:
            return self._sanitize_state(self.recycle_buffer[0])
        return self._sanitize_state(self.recycle_buffer[-self.recycle_delay_steps - 1])

    def _sanitize_state(self, x, fallback=None):
        x = np.asarray(x, dtype=float).copy()
        if fallback is None:
            fallback = self.xs
        fallback = np.asarray(fallback, dtype=float)
        x = np.where(np.isfinite(x), x, fallback)
        return np.clip(x, self.state_low, self.state_high)

    def _generate_noise_impl(self, rng, N, noise_type="gaussian", level=None):
        T = int(N * self.sampling_steps)
        x_dim = int(self.x_dim)

        if noise_type == "gaussian":
            if level is None:
                base_std = np.asarray(self.kw)
                noise = rng.normal(loc=0.0, scale=base_std, size=(T, x_dim))
                dynamic_bw = np.asarray(self.bw)
            else:
                pct = float(level)
                base_amp = np.abs(self.xs)
                scale = pct * base_amp
                noise = rng.normal(loc=0.0, scale=scale, size=(T, x_dim))
                dynamic_bw = 3.0 * scale
            noise = np.clip(noise, -dynamic_bw, dynamic_bw)
        elif noise_type == "sine":
            alpha = float(level) if level is not None else 0.1
            t = np.arange(T, dtype=np.float32) * self.h
            sine = alpha * np.sin(10.0 * np.pi * t)[:, None]
            base_amp = np.abs(self.xs)
            noise = sine * base_amp[None, :]
            dynamic_bw = 3.0 * (alpha * base_amp) if level is not None else np.asarray(self.bw)
            noise = np.clip(noise, -dynamic_bw, dynamic_bw)
        elif noise_type == "uniform":
            epsilon = float(level) if level is not None else 0.1
            base_amp = np.abs(self.xs)
            noise = rng.uniform(low=-epsilon, high=epsilon, size=(T, x_dim))
            noise = noise * base_amp[None, :]
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Choose from ['gaussian','sine','uniform'].")

        return np.asarray(noise)

    def generate_noise(self, N, noise_type="gaussian", level=None):
        return self._generate_noise_impl(np.random, N, noise_type=noise_type, level=level)

    def generate_noise_with_rng(self, rng, N, noise_type="gaussian", level=None):
        return self._generate_noise_impl(rng, N, noise_type=noise_type, level=level)

    def derivative(self, x, us, x_tau=None):
        x = self._sanitize_state(x)
        x_tau = x if x_tau is None else self._sanitize_state(x_tau)

        xA1 = x[0]
        xB1 = x[1]
        T1 = max(x[2], self.state_low[2])
        xA2 = x[3]
        xB2 = x[4]
        T2 = max(x[5], self.state_low[5])
        xA3 = x[6]
        xB3 = x[7]
        T3 = max(x[8], self.state_low[8])

        xA3_tau = x_tau[6]
        xB3_tau = x_tau[7]
        T3_tau = max(x_tau[8], self.state_low[8])

        Q1 = us[0]
        Q2 = us[1]
        Q3 = us[2]

        xC3 = 1 - xA3 - xB3
        x3a = max(self.aA * xA3 + self.aB * xB3 + self.aC * xC3, self._eps)
        xAr = self.aA * xA3 / x3a
        xBr = self.aB * xB3 / x3a
        xCr = self.aC * xC3 / x3a

        xC3_tau = 1 - xA3_tau - xB3_tau
        x3a_tau = max(self.aA * xA3_tau + self.aB * xB3_tau + self.aC * xC3_tau, self._eps)
        xAr_tau = self.aA * xA3_tau / x3a_tau
        xBr_tau = self.aB * xB3_tau / x3a_tau

        F1 = self.F10 + self.Fr
        F2 = F1 + self.F20

        f1 = (
            self.F10 * (self.xA10 - xA1) / self.V1
            + self.Fr * (xAr_tau - xA1) / self.V1
            - self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1
        )
        f2 = (
            self.F10 * (self.xB10 - xB1) / self.V1
            + self.Fr * (xBr_tau - xB1) / self.V1
            + self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1
            - self.k2 * np.exp(-self.E2 / (self.R * T1)) * xB1
        )
        f3 = (
            self.F10 * (self.T10 - T1) / self.V1
            + self.Fr * (T3_tau - T1) / self.V1
            - self.dH1 * self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1 / self.Cp
            - self.dH2 * self.k2 * np.exp(-self.E2 / (self.R * T1)) * xB1 / self.Cp
            + Q1 / (self.rho * self.Cp * self.V1)
        )

        f4 = F1 * (xA1 - xA2) / self.V2 + self.F20 * (self.xA20 - xA2) / self.V2 - self.k1 * np.exp(
            -self.E1 / (self.R * T2)
        ) * xA2
        f5 = (
            F1 * (xB1 - xB2) / self.V2
            + self.F20 * (self.xB20 - xB2) / self.V2
            + self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2
            - self.k2 * np.exp(-self.E2 / (self.R * T2)) * xB2
        )
        f6 = (
            F1 * (T1 - T2) / self.V2
            + self.F20 * (self.T20 - T2) / self.V2
            - self.dH1 * self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2 / self.Cp
            - self.dH2 * self.k2 * np.exp(-self.E2 / (self.R * T2)) * xB2 / self.Cp
            + Q2 / (self.rho * self.Cp * self.V2)
        )

        f7 = F2 * (xA2 - xA3) / self.V3 - (self.Fr + self.Fp) * (xAr - xA3) / self.V3
        f8 = F2 * (xB2 - xB3) / self.V3 - (self.Fr + self.Fp) * (xBr - xB3) / self.V3
        f9 = (
            F2 * (T2 - T3) / self.V3
            + Q3 / (self.rho * self.Cp * self.V3)
            + (self.Fr + self.Fp)
            * (xAr * self.Hvap1 + xBr * self.Hvap2 + xCr * self.Hvap3)
            / (self.rho * self.Cp * self.V3)
        )

        F = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9])
        return np.where(np.isfinite(F), F, 0.0)

    def render(self, mode="human"):
        return

    def get_action(self, noise=True):
        if self.t % self.action_sample_period == 0:
            self.a_holder = self.us
        a = self.a_holder + np.random.normal(np.zeros_like(self.us), self.us * 0.01) if noise else self.a_holder
        return np.clip(a, self.action_low, self.action_high)

    def get_action_pid(self):
        if len(self.state_buffer.memory_action) == 0:
            action = self.us.copy()
        else:
            action = self.state_buffer.memory_action[-1].copy()

        yk = self.state_buffer.memory[-1][self.observed_dims]
        if len(self.state_buffer.memory) == 1:
            yk_1 = self.ys.copy()
        else:
            yk_1 = self.state_buffer.memory[-2][self.observed_dims].copy()

        K_1 = 90
        Ki_1 = 0.004
        ek_1 = self.ys[0] - yk[0]
        action[0] = action[0] + K_1 * (ek_1 + Ki_1 * ek_1 * self.sampling_period)

        K_2 = 70
        Ki_2 = 0.002
        ek_2 = self.ys[1] - yk[1]
        action[1] = action[1] + K_2 * (ek_2 + Ki_2 * ek_2 * self.sampling_period)

        K_3 = 100
        Ki_3 = 0.004
        ek_3 = self.ys[2] - yk[2]
        action[2] = action[2] + K_3 * (ek_3 + Ki_3 * ek_3 * self.sampling_period)

        return np.clip(action, self.action_low, self.action_high)

    def get_noise(self):
        scale = 0.1 * self.xs
        return np.random.normal(np.zeros_like(self.xs), scale)


class state_buffer(object):
    def __init__(self, delay):
        self.delay = delay
        self.memory = []
        self.memory_action = []

    def memorize(self, s):
        self.memory.append(s)

    def memorize_u(self, u):
        self.memory_action.append(u)

    def get_state(self, t):
        if t < self.delay:
            return None
        return self.memory[t]

    def get_action(self, t):
        if t < self.delay:
            return None
        return self.memory_action[t]

    def reset(self):
        self.memory = []
        self.memory_action = []
