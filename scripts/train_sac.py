import gymnasium as gym
from stable_baselines3 import SAC
import highway_env
from matplotlib import pyplot as plt
import os
import numpy as np

env = gym.make("parallel-parking-v0", render_mode="rgb_array")
tensorboard_log = "sac_parking_log/"
os.makedirs(tensorboard_log, exist_ok=True)

def train_sac():
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=5000,
        learning_starts=10000,
        batch_size=128,
        tau=0.005,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], qf=[128, 128]),
        ),
        ent_coef='auto',
        target_update_interval=1,
        target_entropy='auto',
        use_sde=False,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device='auto',
    )
    
    model.learn(total_timesteps=500_000, log_interval=4, progress_bar=True)
    model.save("sac_parallel_model")
    return model

def evaluate_sac():
    model = SAC.load("sac_parallel_model")
    obs, info = env.reset()
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    print("Training SAC...")
    train_sac()
    print("Training completed! Starting evaluation...")
    evaluate_sac()