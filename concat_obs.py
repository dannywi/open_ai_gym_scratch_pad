from collections import deque
from pickletools import uint8
import gym
from gym import spaces
import numpy as np
from typing import Any
from typing import Tuple

class ConcatObs(gym.Wrapper):
  def __init__(self, env: gym.Env, k: uint8) -> None:
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    obs_space_shape = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(
        (k,) + obs_space_shape), dtype=env.observation_space.dtype)

  def reset(self) -> np.array:
    obs = self.env.reset()
    for _ in range(self.k):
      self.frames.append(obs)
    return self._get_ob()

  def step(self, action: Any) -> Tuple[np.array, float, bool, bool]:
    obs, reward, done, info, _ = self.env.step(action)
    self.frames.append(obs)
    return (self._get_ob(), reward, done, info)

  def _get_ob(self) -> np.array:
    return np.array(self.frames)
