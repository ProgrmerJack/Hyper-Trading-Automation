"""Reinforcement-learning helpers for dynamic leverage selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from gym import Env, spaces
except Exception:  # pragma: no cover
    Env = object  # type: ignore
    spaces = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import DDPG
except Exception:  # pragma: no cover
    DDPG = None  # type: ignore


@dataclass
class EnvConfig:
    leverage_range: Tuple[int, int] = (10, 100)


class CustomTradingEnv(Env):
    """Minimal trading environment exposing leverage as an action."""

    metadata: dict[str, list[str]] = {"render.modes": []}

    def __init__(self, config: EnvConfig | None = None):  # pragma: no cover - simple container
        self.config = config or EnvConfig()
        low, high = self.config.leverage_range
        if spaces is not None:
            self.action_space = spaces.Box(
                low=np.array([low], dtype=np.float32),
                high=np.array([high], dtype=np.float32),
            )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        self.state = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # pragma: no cover - deterministic
        self.state = np.zeros_like(self.state)
        return self.state, {}

    def step(self, action):  # pragma: no cover - simplistic dynamics
        leverage = float(action[0]) if np.ndim(action) else float(action)
        reward = -abs(leverage) * 0.001  # discourage large leverage by default
        return self.state, reward, False, False, {}


def train_leverage_agent(env: CustomTradingEnv, timesteps: int = 1000):
    """Train a DDPG agent to optimise leverage.

    The function requires :mod:`stable_baselines3` and will raise an
    :class:`ImportError` if the package is not installed.  The returned
    model can be used to predict an optimal leverage value for a given
    observation from :class:`CustomTradingEnv`.
    """

    if DDPG is None:  # pragma: no cover - optional dependency
        raise ImportError("stable-baselines3 is required for RL training")

    model = DDPG("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model
