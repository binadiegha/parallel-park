"""
Train PPO Agent on Parallel Parking Environment
"""
import sys
import os
import math
# Add the parent directory to Python path to find parallel_parking_env
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import gymnasium as gym
import advanced_parallel_parking_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import numpy as np
from gymnasium.wrappers import TimeLimit

# ===========================
# CONFIG
# ===========================
ENV_ID = "advanced-parallel-parking-v0"
TOTAL_TIMESTEPS = 1_000_000  # Increased training time
N_ENVS = 4  # Increased number of environments for parallel training
LOG_DIR = "./ppo_logs/"
MODEL_PATH = "./models/ppo_parking"

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ===========================
# SETUP
# ===========================
print("🔧 Setting up training environment...")

def make_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")  # 👈 Ensure render_mode is set
    # Remove TimeLimit wrapper if it exists
    while isinstance(env, TimeLimit):  
        env = env.env
    env = Monitor(env, LOG_DIR)
    return env

env = DummyVecEnv([make_env for _ in range(N_ENVS)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

# Improved learning rate schedule with exponential decay
def lr_schedule(progress_remaining: float) -> float:
    return 3e-4 * math.exp(-3.0 * (1 - progress_remaining))
    
# Initialize model with updated hyperparameters
model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=lr_schedule,
    ent_coef=0.2,  # Reduced entropy coefficient
    clip_range=0.2,  # Reduced clip range
    gamma=0.99,  # Increased gamma for longer horizon
    gae_lambda=0.95,
    max_grad_norm=0.5,  # Reduced gradient norm
    normalize_advantage=True,
    n_steps=2048,
    batch_size=64,  # Reduced batch size
    target_kl=0.05,  # Reduced target KL
    verbose=1,
    device='cuda',
    tensorboard_log="./tensorboard_logs",
    policy_kwargs=dict(
        log_std_init=0.5,  # Slightly higher initial std for exploration
        ortho_init=True,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
)

# Enhanced callback with episode length and success rate
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_lengths = []
        self.successes = []

    def _on_step(self) -> bool:
        # Collect episode info
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if "l" in ep_info:  # episode length
                    self.episode_lengths.append(ep_info["l"])
                if "is_success" in ep_info:  # from your env's info
                    self.successes.append(ep_info["is_success"])

        # Log every 1000 steps
        if self.n_calls % 1000 == 0:
            if self.episode_lengths:
                mean_length = np.mean(self.episode_lengths[-100:])  # last 100
                self.logger.record("train/ep_len_mean", mean_length)
            if self.successes:
                success_rate = np.mean(self.successes[-100:])
                self.logger.record("train/success_rate", success_rate)

        return True

# Evaluation callback with more frequent evaluation
eval_callback = EvalCallback(
    env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=2500,  # More frequent evaluation
    n_eval_episodes=5,  # More evaluation episodes
    deterministic=True,
    render=False
)

# Setup logging
logger = configure(LOG_DIR, ["stdout", "tensorboard"])
model.set_logger(logger)

# ===========================
# TRAIN
# ===========================
print(f"🚀 Starting PPO training for {TOTAL_TIMESTEPS:,} timesteps...")
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, TensorboardCallback()],
        tb_log_name="PPO_Parking",
        progress_bar=True,
        reset_num_timesteps=False  # Continue from previous training
    )
except KeyboardInterrupt:
    print("⚠️ Training interrupted by user")
    
# ===========================
# SAVE
# ===========================
model.save(MODEL_PATH + "_final")
env.save(MODEL_PATH + "_vec_normalize.pkl")
print(f"✅ Training complete! Model saved to {MODEL_PATH}_final.zip")