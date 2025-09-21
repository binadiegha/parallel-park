from collections import OrderedDict
import functools

import elements
import embodied
import gymnasium as gym
import numpy as np


class FromGym(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      self._env = gym.make(env, **kwargs)
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    # obs, reward, self._done, self._info = self._env.step(action) # old way for gym
    
    step_result = self._env.step(action)
    if len(step_result) == 5:
      obs, reward, terminated, truncated, self._info = step_result
      self._done = terminated or truncated
      
    elif len(step_result) == 4:
      obs, reward, self._done, self._info = step_result
    
    else:
      raise ValueError(f"Unexpected step return format: {len(step_result)} values")
      
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  # def _obs(
  #     self, obs, reward, is_first=False, is_last=False, is_terminal=False):
  #   if not self._obs_dict:
  #     obs = {self._obs_key: obs}
  #   obs = self._flatten(obs)
  #   obs = {k: np.asarray(v) for k, v in obs.items()}
  #   obs.update(
  #       reward=np.float32(reward),
  #       is_first=is_first,
  #       is_last=is_last,
  #       is_terminal=is_terminal)
  #   return obs
  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    # Handle different observation formats
    if isinstance(obs, tuple):
        # If it's a tuple, try to extract the observation part
        if len(obs) == 2 and isinstance(obs[0], (dict, OrderedDict)):
            obs = obs[0]  # Take the observation, ignore info
        else:
            # If it's some other tuple format, flatten it
            obs = np.concatenate([np.asarray(x).flatten() for x in obs])
    
    if isinstance(obs, np.ndarray):
        # If observation is already flattened to a numpy array
        obs_dict = {'observation': obs}
    elif isinstance(obs, (dict, OrderedDict)):
        # If it's still a dictionary, convert to arrays
        obs_dict = {k: np.asarray(v) for k, v in obs.items()}
    else:
        # Fallback: convert whatever it is to an array
        obs_dict = {'observation': np.asarray(obs)}
    
    # Add standard fields
    obs_dict.update({
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    })
    
    return obs_dict

  def render(self):
    image = self._env.render('rgb_array')
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  # def _flatten(self, nest, prefix=None):
  #   result = {}
  #   for key, value in nest.items():
  #     key = prefix + '/' + key if prefix else key
  #     if isinstance(value, gym.spaces.Dict):
  #       value = value.spaces
  #     if isinstance(value, dict):
  #       result.update(self._flatten(value, key))
  #     else:
  #       result[key] = value
  #   return result
  
  def _flatten(self, nest):
    if isinstance(nest, tuple):
        # Handle the case where a tuple is passed instead of dict
        if len(nest) == 2 and isinstance(nest[0], (dict, OrderedDict)):
            # Likely (obs, info) tuple from reset
            nest = nest[0]
        else:
            return nest
    
    if isinstance(nest, (dict, OrderedDict)):
        # Check if this is observation space definition or actual observation data
        first_value = next(iter(nest.values()))
        
        # If values are gymnasium spaces, preserve structure (for obs_space)
        if hasattr(first_value, 'sample'):  # It's a gymnasium space
            return nest
        
        # If values are numpy arrays, flatten them (for actual observations)
        else:
            result = {}
            for key, value in nest.items():
                if isinstance(value, np.ndarray):
                    result[key] = value.flatten()
                else:
                    result[key] = np.array(value).flatten()
            return np.concatenate(list(result.values()))
    
    return nest

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)
