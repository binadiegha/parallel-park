import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import imageio
import cv2  # 👈 Add this for frame resizing
import advanced_parallel_parking_env

def evaluate_ppo_agent(
    model_path: str = "./models/ppo_parking_final.zip",
    env_id: str = "advanced-parallel-parking-v0",
    n_episodes: int = 5,
    render: bool = False,
    record_videos: bool = True,
    video_path: str = "./videos",
    deterministic: bool = False
) -> dict:
    """
    Evaluate a trained PPO agent with proper video recording.
    """
    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create video directory
    if record_videos:
        os.makedirs(video_path, exist_ok=True)

    # Create environment
    env = gym.make(env_id, render_mode="rgb_array")
    env = Monitor(env)
    
    # Load model
    model = PPO.load(model_path)

    # Metrics
    total_rewards = []
    episode_lengths = []
    successes = []
    video_paths = []

    print(f"Evaluating for {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        frames = [] if record_videos else None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Record frame for video
            if record_videos:
                frame = env.render()
                # Resize frame to be divisible by 16 (fixes ffmpeg warning)
                h, w = frame.shape[:2]
                new_h = ((h + 15) // 16) * 16
                new_w = ((w + 15) // 16) * 16
                frame_resized = cv2.resize(frame, (new_w, new_h))
                frames.append(frame_resized)

            # Render if requested
            if render:
                env.render()

            # Check if episode is done
            if done or truncated:
                # Check for success
                success = info.get("is_success", False)
                successes.append(success)
                total_rewards.append(episode_reward)
                episode_lengths.append(steps)
                
                print(f"Episode {ep+1}/{n_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Steps: {steps} | "
                      f"Success: {success}")

                # Save video
                if record_videos and frames:
                    video_file = os.path.join(video_path, f"episode_{ep+1}.mp4")
                    imageio.mimsave(video_file, frames, fps=10)
                    video_paths.append(video_file)
                    print(f"Saved video to {video_file}")

                break  # 👈 EXIT THE WHILE LOOP — episode is over

    # Calculate metrics
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = np.mean(successes)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Success Rate: {success_rate:.1%} ({sum(successes)}/{n_episodes})")
    print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("="*50)

    # Close environment
    env.close()

    return {
        "success_rate": float(success_rate),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "video_paths": video_paths
    }


if __name__ == "__main__":
    results = evaluate_ppo_agent(
        model_path="./models/ppo_parking_final.zip",
        n_episodes=5,
        render=False,
        record_videos=True
    )