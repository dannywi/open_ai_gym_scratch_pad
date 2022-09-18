import gym
import time
from concat_obs import ConcatObs

wrapped_env = ConcatObs(gym.make("BreakoutNoFrameskip-v4", render_mode="human"), 4)

# observation and action space
print(f"observation space: {wrapped_env.observation_space}")
print(f"action space: {wrapped_env.action_space}")
# help(wrapped_env.unwrapped)
print(f"{wrapped_env.unwrapped.get_action_meanings()}")

# actions to loop
actions_list = wrapped_env.unwrapped.get_action_meanings()
LEFT = actions_list.index('LEFT')
RIGHT = actions_list.index('RIGHT')
FIRE = actions_list.index('FIRE')
actions = [FIRE] * 1 + [RIGHT] * 30 + [LEFT] * 30

obs = wrapped_env.reset()

for i in range(500):
  action = actions[i % len(actions)]
  # print(f"action: {action}")
  obs, reward, done, info = wrapped_env.step(action)
  wrapped_env.render()
  time.sleep(0.002)

wrapped_env.close()
