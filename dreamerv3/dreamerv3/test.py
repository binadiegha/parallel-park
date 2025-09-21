# # Test this first:
# import sys
# sys.path.append('./highway_env')

# import highway_env
# import gymnasium as gym

# try:
#     env = gym.make('parallel-parking')
#     print("Environment created successfully!")
#     obs, info = env.reset()
#     print(f"Obs type: {type(obs)}")
#     env.close()
# except Exception as e:
#     print(f"Error: {e}")

import highway_env
import gymnasium as gym

env = gym.make('parallel-parking-v0')
obs, info = env.reset()
print(f"Reset obs type: {type(obs)}")
print(f"Reset obs keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")

action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
print(f"Step obs type: {type(obs)}")
print(f"Step obs keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")