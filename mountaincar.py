import gym
import time
import random

env = gym.make("MountainCar-v0", render_mode="human")

# observation and action space
print(f"observation space: {env.observation_space}")
print(f"action space: {env.action_space}")

# reset environment and see initial observation
obs = env.reset()
last_obs = None
print(f"initial observation: {obs}")

action = random.choice([0, 2]) # select left or right
times = 3

LEFT = 0
RIGHT = 2

for _ in range(500):
  # take action and get new observation space
  obs, reward, done, info, _ = env.step(action)

  # LOGIC:
  # to enforce movement, accelerate to left if it's going left and vice versa
  if last_obs is not None and obs[0] > last_obs[0]:
    action = RIGHT
  else:
    action = LEFT

  last_obs = obs

  env.render()
  time.sleep(0.001)

  if done:
    print("DONE")
    env.reset()
    action = random.choice([0, 2])
    times -= 1
    if times == 0:
      break

env.close()