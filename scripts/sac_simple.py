import gymnasium as gym

from stable_baselines3 import SAC
import highway_env
from matplotlib import pyplot as plt
import os
# import dreamerv3
import numpy as np
import enum 

venv = os.environ.get("CONDA_DEFAULT_ENV")
if venv == "dreamer":
    import dreamerv3

env = gym.make("parallel-parking-v0", render_mode="rgb_array")

    
tensorboard_log="sac_parking_log/"
os.makedirs(tensorboard_log, exist_ok=True)

train = True
training_model = "SAC" #"SAC"

def train_model():
    # override env config to work with SAC
    # env.unwrapped.config.update({
    #     "observation": {
    #         "type": "KinematicsGoal",
    #         "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
    #         "scales": [100, 100, 5, 5, 1, 1],
    #         "normalize": False,
    #     },
    #     "action": {"type": "ContinuousAction"},

    #  })
     
    model = SAC(
            "MultiInputPolicy",  # Multi-layer perceptron for continuous observations MlpPolicy
            env,
            
            # SAC-specific hyperparameters
            learning_rate=3e-4,           # Standard learning rate for SAC
            buffer_size=5000,            # Reasonable replay buffer size
            learning_starts=10000,         # Start learning after some experience
            batch_size=128,               # Good batch size for this problem
            tau=0.005,                    # Soft update coefficient
            gamma=0.95,                   # Discount factor (higher for longer episodes)
            
            # Training frequency
            train_freq=1,                 # Train after every step
            gradient_steps=1,             # One gradient step per training step
            
            # Network architecture
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], qf=[128, 128]),  # Reasonable network size
            ),
            
            # SAC-specific parameters
            ent_coef='auto',              # Automatic entropy coefficient tuning
            target_update_interval=1,     # Update target networks every step
            target_entropy='auto',        # Automatic target entropy
            use_sde=False,                # No state-dependent exploration
            
            # Logging
            verbose=1,
            tensorboard_log=tensorboard_log,
            device='auto',                # Use GPU if available
        )

    model.learn(total_timesteps=500_000, log_interval=4, progress_bar=True)
    model.save("sac_parallel_model")

    del model # remove to demonstrate saving and loading


def train_dreamer():
    
    # visual learning 
    #     "observation": {
    #     "type": "GrayscaleObservation",
    #     "observation_shape": (64, 64),
    #     "stack_size": 4,
    # },
    # "normalize_reward": True
    # update config to work with dreamer
    env.unwrapped.config.update({
        # configuration for dreamerv3 env
            "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False,
    },
    "normalize_reward": True,
    "reward_clip": 5.0
    
    })
    print(dreamerv3)
    
# dreamer model ends here
if train:
    if training_model == "dreamer":
        train_dreamer()
    else:    
        train_model()
    
model = SAC.load("sac_parallel_model")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
        
# plt.imshow(env.render())
# plt.show()
        
        
