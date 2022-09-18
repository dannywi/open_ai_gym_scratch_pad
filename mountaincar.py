import gym
import time
import random
from concat_obs import ConcatObs
import numpy as np

env = ConcatObs(gym.make("MountainCar-v0", render_mode="human"), 2)

# observation and action space
print(f"observation space: {env.observation_space}")
print(f"action space: {env.action_space}")

LEFT = 0
RIGHT = 2

for r in range(1, 4):
  # reset environment and see initial observation
  obs = env.reset()
  print(f"initial observation: {obs}")
  action = random.choice([LEFT, RIGHT])

  for _ in range(500):
    # take action and get new observation space
    obs, reward, done, info = env.step(action)

    # LOGIC:
    # to enforce movement, accelerate to left if it's going left and vice versa
    # FIX: after initialization, somehow each obs's first elem is not the X axis
    if type(obs[0][0]) is np.float32 and obs[1][0] > obs[0][0]:
      action = RIGHT
    else:
      action = LEFT

    env.render()
    time.sleep(0.001)

    if done:
      print(f"DONE RUN {r}")
      break

env.close()
