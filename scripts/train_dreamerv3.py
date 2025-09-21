import gymnasium as gym
import highway_env
import os
import numpy as np

# Only import dreamerv3 in this file
try:
    import dreamerv3
    DREAMER_AVAILABLE = True
except ImportError:
    print("DreamerV3 not available. Please install it first.")
    DREAMER_AVAILABLE = False
    exit(1)

def setup_dreamer_env():
    """Setup environment with DreamerV3-specific configuration"""
    env = gym.make("parallel-parking-v0", render_mode="rgb_array")
    
    # Update config for DreamerV3
    env.unwrapped.config.update({
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        },
        "normalize_reward": True,
        "reward_clip": 5.0
    })
    
    return env

def get_dreamer_config():
    """DreamerV3 configuration - using embodied.Config"""
    return embodied.Config({
        # Logging
        'logdir': './dreamer_logs',
        'run.script': 'train',
        'run.steps': 500_000,
        'run.eval_every': 10_000,
        'run.save_every': 50_000,
        
        # Environment
        'task': 'parallel_parking',
        'envs.num': 1,
        'envs.parallel': 'none',
        
        # Agent
        'agent': 'dreamerv3',
        
        # Batch settings
        'batch_size': 16,
        'batch_length': 64,
        
        # Learning rates
        'agent.actor_lr': 3e-5,
        'agent.critic_lr': 3e-5,
        'agent.world_lr': 1e-4,
        
        # Model architecture
        'agent.rssm.units': 512,
        'agent.rssm.stoch': 32,
        'agent.rssm.deter': 512,
        
        # World model
        'agent.world_model.grad_heads': ['decoder', 'reward', 'cont'],
        'agent.world_model.pred_discount': True,
        
        # Replay buffer
        'replay.size': 1e6,
        'replay.online': False,
    })

def train_dreamer():
    """Train DreamerV3 agent"""
    print("Setting up DreamerV3 training...")
    
    # Setup environment and config
    env = setup_dreamer_env()
    config = get_dreamer_config()
    
    print("Starting DreamerV3 training...")
    
    # Create embodied environment wrapper
    def make_env():
        return embodied.envs.FromGym(env)
    
    # Create the training setup
    try:
        # Method 1: Use embodied framework (most common for DreamerV3)
        embodied.run.train(make_env, config)
        
    except Exception as e:
        print(f"Embodied training failed: {e}")
        print("Trying alternative approach...")
        
        # Method 2: Direct agent usage (if available)
        try:
            agent = dreamerv3.Agent(config)
            
            # Manual training loop
            obs = env.reset()[0]
            step = 0
            episode_reward = 0
            
            while step < config['run.steps']:
                # Convert observation format if needed
                if isinstance(obs, dict):
                    # Handle goal-conditioned observations
                    obs_array = np.concatenate([
                        obs['observation'],
                        obs['achieved_goal'], 
                        obs['desired_goal']
                    ])
                else:
                    obs_array = obs
                
                # Get action from agent
                action = agent.policy(obs_array)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update episode tracking
                episode_reward += reward
                step += 1
                
                if done:
                    print(f"Step {step}: Episode reward: {episode_reward:.2f}")
                    episode_reward = 0
                    obs = env.reset()[0]
                else:
                    obs = next_obs
                
                # Periodic evaluation
                if step % config['run.eval_every'] == 0:
                    print(f"Training step: {step}")
            
        except Exception as e2:
            print(f"Direct agent approach also failed: {e2}")
            print("Please check your DreamerV3 installation and API.")
            return None
    
    print("DreamerV3 training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DreamerV3 Training for Parallel Parking')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                       help='Training or evaluation mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_dreamer()
    else:
        print("Evaluation mode not implemented yet")