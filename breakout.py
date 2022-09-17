import gym
import time

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")

# observation and action space
print(f"observation space: {env.observation_space}")
print(f"action space: {env.action_space}")
#help(env.unwrapped)
print(f"{env.unwrapped.get_action_meanings()}")

obs = env.reset()

LEFT = 3
RIGHT = 2
FIRE = 1

actions = [FIRE] * 1 + [RIGHT] * 30 + [LEFT] * 30

for i in range(500):
  action = actions[i % len(actions)]
  print(f"action: {action}")
  obs, reward, done, info, _ = env.step(action)
  env.render()
  time.sleep(0.02)

env.close()