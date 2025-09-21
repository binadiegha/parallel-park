import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple

# Import the embodied framework components
try:
    from . import from_gym
except ImportError:
    # Handle case where from_gym might not exist or be structured differently
    from_gym = None

# Import highway_env to register environments
import highway_env
from highway_env.envs.parallel_parking_env import ParallelParkingEnv

# Register your custom environment
gym.register(
    id='ParallelParkingEnv-v0',
    entry_point='highway_env.envs.parallel_parking_env:ParallelParkingEnv',
)


class HighwayEnv:
    """Highway-env wrapper for the embodied framework"""
    
    def __init__(self, name, **kwargs):
        # Create the gym environment
        if name == 'parallel_parking':
            self._env = gym.make('ParallelParkingEnv-v0', render_mode=None)
        elif name == 'ParallelParkingEnv-v0':
            self._env = gym.make('ParallelParkingEnv-v0', render_mode=None)
        else:
            # Support other highway-env environments
            self._env = gym.make(name, render_mode=None)
        
        # Handle goal-based environments by flattening observations
        self._is_goal_env = hasattr(self._env, 'compute_reward')
        
        if self._is_goal_env:
            # Get sample observation to understand structure
            sample_obs, _ = self._env.reset()
            if isinstance(sample_obs, dict):
                # Calculate flattened dimension
                total_dim = 0
                self._obs_keys = []
                for key in ['observation', 'achieved_goal', 'desired_goal']:
                    if key in sample_obs:
                        self._obs_keys.append(key)
                        total_dim += np.prod(sample_obs[key].shape)
                
                # Create flat observation space
                self._obs_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(total_dim,),
                    dtype=np.float32
                )
            else:
                self._obs_space = self._env.observation_space
        else:
            self._obs_space = self._env.observation_space
        
        self._act_space = self._env.action_space
        
        # Reset environment to initial state
        self._obs, _ = self._env.reset()
        self._done = True
    
    def _flatten_obs(self, obs):
        """Convert goal-based dict observation to flat array"""
        if self._is_goal_env and isinstance(obs, dict):
            parts = []
            for key in self._obs_keys:
                if key in obs:
                    part = obs[key]
                    if not isinstance(part, np.ndarray):
                        part = np.array(part)
                    parts.append(part.flatten())
            
            if parts:
                return np.concatenate(parts).astype(np.float32)
            else:
                # Fallback
                return np.array(list(obs.values())).flatten().astype(np.float32)
        else:
            return np.array(obs).astype(np.float32)
    
    @property
    def obs_space(self):
        return self._obs_space
    
    @property
    def act_space(self):
        return self._act_space
    
    def step(self, action):
        """Take environment step"""
        # Convert action to proper format
        if isinstance(action, dict) and 'action' in action:
            action = action['action']
        
        if isinstance(action, np.ndarray):
            action = action.astype(np.float32)
        
        # Take step
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        
        # Store for next step
        self._obs = self._flatten_obs(obs)
        self._done = done
        
        # Return in embodied framework format
        return {
            'obs': self._obs,
            'reward': np.float32(reward),
            'is_first': False,
            'is_last': done,
            'is_terminal': terminated,
            **info
        }
    
    def reset(self):
        """Reset environment"""
        obs, info = self._env.reset()
        self._obs = self._flatten_obs(obs)
        self._done = False
        
        return {
            'obs': self._obs,
            'reward': np.float32(0.0),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            **info
        }
    
    def render(self):
        """Render environment"""
        return self._env.render()
    
    def close(self):
        """Close environment"""
        if hasattr(self, '_env'):
            self._env.close()


# Factory function for embodied framework
def make_highway_env(name, **kwargs):
    """Create highway environment compatible with embodied framework"""
    env = HighwayEnv(name, **kwargs)
    return env


# Register environments
HIGHWAY_ENVS = {
    'parallel_parking': make_highway_env,
    'ParallelParkingEnv-v0': make_highway_env,
    # Add other highway environments as needed
    'highway-v0': make_highway_env,
    'highway-fast-v0': make_highway_env,
    'merge-v0': make_highway_env,
    'roundabout-v0': make_highway_env,
    'parking-v0': make_highway_env,
    'intersection-v0': make_highway_env,
    'racetrack-v0': make_highway_env,
}