import numpy as np
import gym
from gym import spaces

class KeyMazeWrapper(gym.Wrapper):
    """
    A wrapper for KeyDoorMazeEnv that normalizes the observation space
    to ensure consistent state representations for baseline agents.
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper with the environment.
        
        Args:
            env: The KeyDoorMazeEnv to wrap
        """
        super(KeyMazeWrapper, self).__init__(env)
        
        # Define the observation space based on the maze dimensions
        maze_size = env.maze_size
        # KeyDoorMazeEnv has 5 channels: empty, wall, goal, key, door, agent position
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(5, maze_size, maze_size),
            dtype=np.float32
        )
    
    def reset(self):
        """Reset the environment and normalize the state."""
        state = self.env.reset()
        return self._normalize_state(state)
    
    def step(self, action):
        """Take a step in the environment and normalize the resulting state."""
        if isinstance(action, np.ndarray) and action.shape == ():
            # Convert numpy scalar to int
            action = action.item()
            
        # Call the environment's step method
        result = self.env.step(action)
        
        # Handle different result formats (old vs new gym API)
        if len(result) == 5:  # New Gym API: next_state, reward, done, truncated, info
            next_state, reward, done, truncated, info = result
            # Normalize the next state
            normalized_state = self._normalize_state(next_state)
            return normalized_state, reward, done, truncated, info
        else:  # Old Gym API: next_state, reward, done, info
            next_state, reward, done, info = result
            # Normalize the next state
            normalized_state = self._normalize_state(next_state)
            return normalized_state, reward, done, info
    
    def _normalize_state(self, state):
        """
        Normalize the state from KeyDoorMazeEnv to a consistent format.
        
        Args:
            state: Raw state from the environment
            
        Returns:
            Normalized state as a numpy array with shape (5, maze_size, maze_size)
        """
        # If the state is already a numpy array with the right shape, return it
        if isinstance(state, np.ndarray) and len(state.shape) == 3:
            # Convert from (H, W, C) to (C, H, W) if needed
            if state.shape[2] == 5:
                return np.transpose(state, (2, 0, 1)).astype(np.float32)
            elif state.shape[0] == 5:
                return state.astype(np.float32)
        
        # Get the state as 5 separate channels from the observation
        try:
            # Check if state is a tuple from the environment
            if isinstance(state, tuple) and len(state) > 0:
                # Extract the main observation (first element)
                state = state[0]
            
            # The image representation might be in (H, W, C) format
            if isinstance(state, np.ndarray) and len(state.shape) == 3:
                if state.shape[2] == 5:  # (H, W, 5)
                    # Convert to (5, H, W) for PyTorch
                    return np.transpose(state, (2, 0, 1)).astype(np.float32)
            
            # If we have the grid directly, convert it to the proper format
            maze_size = self.env.maze_size
            normalized_state = np.zeros((5, maze_size, maze_size), dtype=np.float32)
            
            # Handle different possible state representations
            if hasattr(self.env, 'grid'):
                # Channel 0: Empty spaces (0)
                # Channel 1: Walls (1)
                # Channel 2: Goal (2)
                # Channel 3: Keys (3)
                # Channel 4: Doors (4)
                for i in range(maze_size):
                    for j in range(maze_size):
                        cell_type = self.env.grid[i, j]
                        if 0 <= cell_type <= 4:
                            normalized_state[cell_type, i, j] = 1.0
                
                # Mark agent position in the empty channel
                if hasattr(self.env, 'agent_position'):
                    i, j = self.env.agent_position
                    # Add agent as a highlight in the empty space channel
                    normalized_state[0, i, j] = 0.5
            
            return normalized_state
        
        except Exception as e:
            print(f"Error normalizing state: {e}")
            # Return a default state if normalization fails
            return np.zeros((5, self.env.maze_size, self.env.maze_size), dtype=np.float32)
