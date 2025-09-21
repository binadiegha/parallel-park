import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
import highway_env  # noqa: F401
import os




def train_env():
    """Create training environment with SAC-optimized configuration."""
    # Use your custom environment
    env = gym.make('parallel-parking-compact-v0')
    
    # Override config to disable traffic vehicles (this is the key fix)
    env.unwrapped.config.update({
        # Disable problematic traffic creation
        "vehicles_count": 2,          # Only ego + parked vehicles
        "vehicles_density": 0.0,      # No traffic density
        "other_vehicles_type": None,  # Disable traffic vehicle type
        
        # Training optimizations
        "duration": 40,
        "simulation_frequency": 15,
        "policy_frequency": 8,
        
        # Observation for continuous control (already configured in your env)
        "observation": {
            "type": "KinematicsGoal",
            "vehicles_count": 3,  # ego + 2 parked
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "features_range": {
                "x": [-50, 150],
                "y": [-10, 10], 
                "vx": [-15, 15],
                "vy": [-15, 15],
            },
            "absolute": False,
            "normalize": True,
            "see_behind": True,
        },
        
        # Continuous action space (already configured)
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "dynamical": True,
            "clip": True,
        },
        
        # Training settings
        "show_trajectories": False,
        "offroad_terminal": False,
    })
    
    return env


def test_env():
    """Create test environment with evaluation settings."""
    env = gym.make('parallel-parking-v0')
    
    # Override config
    env.unwrapped.config.update({
        # Disable problematic traffic creation
        "vehicles_count": 2,          # Only ego + parked vehicles
        "vehicles_density": 0.0,      # No traffic density
        "other_vehicles_type": None,  # Disable traffic vehicle type
        
        "duration": 60,
        "simulation_frequency": 20,
        "policy_frequency": 10,
        
        # Observation configuration
        "observation": {
            "type": "KinematicsGoal",
            "vehicles_count": 3,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "features_range": {
                "x": [-50, 150],
                "y": [-10, 10],
                "vx": [-15, 15],
                "vy": [-15, 15],
            },
            "absolute": False,
            "normalize": True,
            "see_behind": True,
        },
        
        # Continuous action space
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "dynamical": True,
            "clip": True,
        },
        
        "show_trajectories": False,
        "offroad_terminal": False,
    })
    
    return env


# Patch the traffic creation method to avoid the error
def patch_traffic_creation():
    """Monkey patch the problematic traffic vehicle creation."""
    
    # def _create_traffic_vehicles_safe(self) -> None:
    #     """Safe version that doesn't create traffic vehicles."""
    #     # Simply do nothing - no traffic vehicles
    #     print("Traffic vehicle creation bypassed for SAC training")
    #     pass
    
    # # Apply the patch to both environment classes
    # ParallelParkingEnv._create_traffic_vehicles = _create_traffic_vehicles_safe
    # ParallelParkingEnvCompact._create_traffic_vehicles = _create_traffic_vehicles_safe


class ParkingEnvWrapper(gym.Wrapper):
    """Wrapper to handle potential issues and add SAC-friendly features."""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.step_count = 0
        self.episode_count = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_reward += reward
        self.step_count += 1
        
        # Add useful info for monitoring
        info.update({
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
        })
        
        # Check if successfully parked using the environment's method
        if hasattr(self.env.unwrapped, '_is_parked_successfully'):
            info['is_success'] = self.env.unwrapped._is_parked_successfully()
        else:
            # Fallback: assume success if reward is high
            info['is_success'] = reward > 0.8
            
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.step_count = 0
        return self.env.reset(**kwargs)


def make_train_env():
    """Make a single training environment with wrapper."""
    try:
        env = train_env()
        env = ParkingEnvWrapper(env)
        return env
    except Exception as e:
        print(f"Error creating training environment: {e}")
        raise


def make_test_env():
    """Make a single test environment."""
    try:
        env = test_env()
        env = ParkingEnvWrapper(env)
        return env
    except Exception as e:
        print(f"Error creating test environment: {e}")
        raise


def create_sac_model(env, tensorboard_log="sac_parking/"):
    """Create SAC model with optimized hyperparameters for parking."""
    
    # Create directories
    
    os.makedirs(tensorboard_log, exist_ok=True)
    
    model = SAC(
        "MlpPolicy",  # Multi-layer perceptron for continuous observations
        env,
        
        # SAC-specific hyperparameters
        learning_rate=3e-4,           # Standard learning rate for SAC
        buffer_size=50000,            # Reasonable replay buffer size
        learning_starts=1000,         # Start learning after some experience
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
    
    return model


def train_sac_parking():
    """Train SAC agent for parallel parking."""
    
    print("Applying traffic vehicle creation patch...")
    patch_traffic_creation()
    
    print("Creating training environment...")
    
    # Test single environment first
    try:
        test_env_creation = make_train_env()
        test_obs, _ = test_env_creation.reset()
        print(f"Environment test successful! Observation shape: {test_obs.shape}")
        test_env_creation.close()
    except Exception as e:
        print(f"Environment creation failed: {e}")
        return None, None
    
    # Create vectorized training environment
    try:
        train_vec_env = make_vec_env(
            make_train_env,
            n_envs=1,  # Start with single environment
            vec_env_cls=DummyVecEnv
        )
        
        # Wrap with monitor for logging
        train_vec_env = VecMonitor(train_vec_env)
        print("Training environment created successfully!")
        
    except Exception as e:
        print(f"Error creating vectorized environment: {e}")
        return None, None
    
    print("Creating evaluation environment...")
    try:
        eval_env = Monitor(make_test_env())
        print("Evaluation environment created successfully!")
    except Exception as e:
        print(f"Warning: Could not create evaluation environment: {e}")
        eval_env = None
    
    print("Initializing SAC model...")
    model = create_sac_model(train_vec_env)
    
    callbacks = []
    
    # Setup evaluation callback if eval env is available
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./sac_parking/best_model/",
            log_path="./sac_parking/eval_logs/",
            eval_freq=2500,               # Evaluate every 2500 steps
            n_eval_episodes=5,            # Use 5 episodes for evaluation
            deterministic=True,           # Use deterministic policy for evaluation
            render=False,
        )
        callbacks.append(eval_callback)
    
    # Setup stop training callback for early stopping
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=0.7,         # Stop when average reward > 0.7
        verbose=1
    )
    callbacks.append(stop_callback)
    
    print("Starting SAC training...")
    print("Training parameters:")
    print(f"  - Total timesteps: {int(5e4):,}")
    print(f"  - Parallel environments: 1")
    print(f"  - Evaluation frequency: every 2,500 steps" if eval_env else "  - No evaluation")
    print(f"  - Early stopping threshold: 0.7 reward")
    
    # Create directories
    os.makedirs("sac_parking", exist_ok=True)
    os.makedirs("sac_parking/best_model", exist_ok=True)
    
    # Train the model
    try:
        model.learn(
            total_timesteps=int(5e4),     # 50k timesteps for initial training
            callback=callbacks if callbacks else None,
            tb_log_name="SAC_parking",
            progress_bar=True,
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training error: {e}")
        print("Saving current model state...")
    
    # Save final model
    model.save("sac_parking/final_model")
    print("Model saved to: sac_parking/final_model")
    
    # Close environments
    if eval_env:
        eval_env.close()
    train_vec_env.close()
    
    return model, train_vec_env


def test_sac_parking(model_path="sac_parking/best_model/best_model"):
    """Test trained SAC agent and record video."""
    
    print("Applying traffic vehicle creation patch...")
    patch_traffic_creation()
    
    print(f"Loading model from: {model_path}")
    
    # Try to load the model, fallback to final model if best model doesn't exist
    try:
        model = SAC.load(model_path)
        print("Loaded best model successfully!")
    except FileNotFoundError:
        print(f"Best model not found, trying final model...")
        try:
            model = SAC.load("sac_parking/final_model")
            print("Loaded final model successfully!")
        except FileNotFoundError:
            print("No trained model found. Please run training first.")
            return
    
    print("Creating test environment for video recording...")
    env = DummyVecEnv([make_test_env])
    
    # Calculate video length based on episode duration
    video_length = env.envs[0].unwrapped.config["duration"] * 2
    
    # Create directories
    os.makedirs("sac_parking/videos", exist_ok=True)
    
    # Setup video recording
    env = VecVideoRecorder(
        env,
        "sac_parking/videos/",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="sac-parking-agent",
    )
    
    print("Recording test episodes...")
    print(f"Video length: {video_length} steps")
    
    obs, info = env.reset()
    episode_rewards = []
    episode_reward = 0
    success_count = 0
    episodes = 0
    
    for step in range(video_length + 1):
        # Use deterministic policy for testing
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, infos = env.step(action)
        
        episode_reward += rewards[0]
        
        # Check for episode end
        if dones[0] or truncated[0]:
            episodes += 1
            episode_rewards.append(episode_reward)
            
            # Check success from info if available
            if len(infos) > 0 and 'is_success' in infos[0]:
                if infos[0]['is_success']:
                    success_count += 1
                    print(f"Episode {episodes}: SUCCESS! Reward: {episode_reward:.3f}")
                else:
                    print(f"Episode {episodes}: Failed. Reward: {episode_reward:.3f}")
            else:
                # Fallback: assume success if reward is high
                if episode_reward > 0.5:
                    success_count += 1
                    print(f"Episode {episodes}: SUCCESS! Reward: {episode_reward:.3f}")
                else:
                    print(f"Episode {episodes}: Failed. Reward: {episode_reward:.3f}")
            
            episode_reward = 0
    
    env.close()
    
    # Print summary
    if episode_rewards:
        print(f"\nTest Results:")
        print(f"  - Episodes completed: {len(episode_rewards)}")
        print(f"  - Success rate: {success_count}/{len(episode_rewards)} ({100*success_count/len(episode_rewards):.1f}%)")
        print(f"  - Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"  - Best reward: {max(episode_rewards):.3f}")
    
    print(f"Video saved to: sac_parking/videos/")


def evaluate_parking_performance(model_path="sac_parking/best_model/best_model", num_episodes=20):
    """Evaluate parking performance over multiple episodes."""
    
    print("Applying traffic vehicle creation patch...")
    patch_traffic_creation()
    
    print(f"Evaluating model performance over {num_episodes} episodes...")
    
    # Try to load the model
    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        try:
            model = SAC.load("sac_parking/final_model")
            print("Using final model for evaluation...")
        except FileNotFoundError:
            print("No trained model found. Please run training first.")
            return None
    
    env = make_test_env()
    
    results = {
        'rewards': [],
        'successes': [],
        'episode_lengths': [],
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 500:  # Safety limit
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        # Determine success
        success = info.get('is_success', episode_reward > 0.5)
        
        results['rewards'].append(episode_reward)
        results['successes'].append(success)
        results['episode_lengths'].append(steps)
        
        if episode % 5 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.3f}, Success={success}, Steps={steps}")
    
    env.close()
    
    # Print comprehensive results
    print(f"\n=== Parking Performance Evaluation ===")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {np.mean(results['successes']):.1%}")
    print(f"Average Reward: {np.mean(results['rewards']):.3f} ± {np.std(results['rewards']):.3f}")
    print(f"Average Episode Length: {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f} steps")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC Parallel Parking Training')
    parser.add_argument('--mode', choices=['train', 'test', 'eval'], default='train',
                       help='Mode: train, test, or eval')
    parser.add_argument('--model_path', type=str, default="sac_parking/best_model/best_model",
                       help='Path to model for testing/evaluation')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting SAC training for parallel parking...")
        model, env = train_sac_parking()
        if model is not None:
            print("Training complete!")
        else:
            print("Training failed!")
        
    elif args.mode == 'test':
        print("Testing SAC agent and recording video...")
        test_sac_parking(args.model_path)
        print("Testing complete!")
        
    elif args.mode == 'eval':
        print("Evaluating SAC agent performance...")
        results = evaluate_parking_performance(args.model_path, args.episodes)
        if results:
            print("Evaluation complete!")
    
    else:
        print("Invalid mode. Use --help for options.")