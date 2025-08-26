"""
Adapter to make the KeyDoorMazeEnv compatible with gymnasium.
"""

import gymnasium as gym
import numpy as np
from key_door_maze_env import KeyDoorMazeEnv

class KeyDoorMazeEnvAdapter(gym.Env):
    """
    Adapter that makes KeyDoorMazeEnv compatible with gymnasium's Env interface.
    This wrapper ensures that the KeyDoorMazeEnv can be used with wrappers that
    expect standard gymnasium.Env interfaces.
    """
    
    def __init__(self, 
                 maze_size=10, 
                 num_keys=1, 
                 max_steps=300,
                 use_fixed_layout=False,
                 use_fixed_seed=False,
                 fixed_seed=42):
        """Initialize the adapter with a KeyDoorMazeEnv."""
        # Create the underlying environment
        self.env = KeyDoorMazeEnv(
            maze_size=maze_size,
            num_keys=num_keys,
            max_steps=max_steps,
            use_fixed_layout=use_fixed_layout,
            use_fixed_seed=use_fixed_seed,
            fixed_seed=fixed_seed
        )
        
        # Define action and observation spaces for gymnasium compatibility
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        
        # Create observation space based on grid size
        # (We'll create a proper observation space as RGB channels)
        # The grid has multiple channels for: agent, walls, goal, keys, doors
        num_channels = 6  # Agent, Wall, Goal, Keys, Doors, Empty
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(self.env.maze_size, self.env.maze_size, num_channels),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed)
        return obs, info
    
    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)
    
    def render(self, mode='rgb_array'):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        return None
    
    def get_coverage_stats(self):
        """Pass through the coverage stats method."""
        if hasattr(self.env, 'get_coverage_stats'):
            return self.env.get_coverage_stats()
        return {'coverage': 0.0, 'visited': 0, 'total': self.env.maze_size * self.env.maze_size}
        
    # Forward all attributes that aren't in the adapter to the wrapped env
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute {name}")
        return getattr(self.env, name)
