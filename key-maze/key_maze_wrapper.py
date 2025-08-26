"""
Key-Door Maze Wrapper for compatibility with Rainbow DCEO networks.

This wrapper adapts the Key-Door Maze environment to be compatible with
the existing agent implementations by reshaping and upscaling the observations.
"""

import numpy as np
import gymnasium as gym
from collections import deque
from key_door_maze_env import KeyDoorMazeEnv
import cv2
import torch

class KeyMazeWrapper(gym.Wrapper):
    """
    Enhanced wrapper for Key-Door Maze environment to handle preprocessing and improved learning signals.
    """
    
    def __init__(self, env, frame_stack=4, resize_shape=(84, 84), proximity_reward=True):
        """Initialize the wrapper."""
        super().__init__(env)
        self.env = env
        self.frame_stack_size = frame_stack
        self.resize_shape = resize_shape
        self.proximity_reward = proximity_reward
        
        # Create frame buffer for frame stacking
        self.frames = deque(maxlen=frame_stack)
        
        # Initialize tracking variables for rewards
        self.previous_keys = 0
        self.previous_doors = 0
        
        # Update observation space to account for frame stacking and resizing
        # We'll use 4 channels for our processed observations
        single_frame_shape = (resize_shape[0], resize_shape[1], 4)
        stacked_frame_shape = (resize_shape[0], resize_shape[1], 4 * frame_stack)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, 
            shape=stacked_frame_shape,
            dtype=np.float32
        )
        
        # Action space remains the same
        self.action_space = env.action_space
        
        # Tracking variables for improved learning signals
        self.previous_keys = 0
        self.previous_doors = 0
        self.previous_min_key_distance = float('inf')
        self.previous_min_door_distance = float('inf')
        self.previous_min_goal_distance = float('inf')
        self.key_positions = []
        self.door_positions = []
        self.goal_position = None
    
    def _preprocess_observation(self, obs):
        """
        Convert multi-channel observation to 4 essential channels for better learning.
        Channel 0: Agent position
        Channel 1: Keys
        Channel 2: Doors
        Channel 3: Walls and goal
        """
        # Debug input observation shape
        print(f"DEBUG: Input observation shape: {obs.shape}")
        
        # Create a 4-channel observation
        processed = np.zeros((self.resize_shape[0], self.resize_shape[1], 4), dtype=np.float32)
        
        # Resize original image while preserving key information
        h, w = obs.shape[0], obs.shape[1]
        
        # Channel 0: Agent (from channel 0)
        agent = cv2.resize(obs[:, :, 0], self.resize_shape, interpolation=cv2.INTER_AREA)
        processed[:, :, 0] = agent
        
        # Channel 1: Keys (from channel 3)
        if 3 < obs.shape[2]:
            keys = cv2.resize(obs[:, :, 3], self.resize_shape, interpolation=cv2.INTER_AREA)
            processed[:, :, 1] = keys
        
        # Channel 2: Doors (from channel 4)
        if 4 < obs.shape[2]:
            doors = cv2.resize(obs[:, :, 4], self.resize_shape, interpolation=cv2.INTER_AREA)
            processed[:, :, 2] = doors
        
        # Channel 3: Walls (from channel 1) and Goal (from channel 2)
        if 1 < obs.shape[2] and 2 < obs.shape[2]:
            walls = cv2.resize(obs[:, :, 1], self.resize_shape, interpolation=cv2.INTER_AREA)
            goal = cv2.resize(obs[:, :, 2], self.resize_shape, interpolation=cv2.INTER_AREA)
            processed[:, :, 3] = walls + goal  # Combine walls and goal
        
        return processed
    
    def _find_object_positions(self, grid):
        """
        Extract positions of keys, doors, and goal from the grid.
        Used for calculating proximity rewards.
        """
        self.key_positions = []
        self.door_positions = []
        self.goal_position = None
        
        # If we have direct access to the grid, use it
        if hasattr(self.env, 'grid'):
            grid = self.env.grid
            size = self.env.maze_size if hasattr(self.env, 'maze_size') else len(grid)
            
            for i in range(size):
                for j in range(size):
                    cell = grid[i, j]
                    # Check if it's a key
                    if hasattr(self.env, 'KEY') and self.env.KEY <= cell < self.env.DOOR:
                        self.key_positions.append((i, j))
                    # Check if it's a door
                    elif hasattr(self.env, 'DOOR') and self.env.DOOR <= cell < self.env.GOAL:
                        self.door_positions.append((i, j))
                    # Check if it's a goal
                    elif hasattr(self.env, 'GOAL') and cell == self.env.GOAL:
                        self.goal_position = (i, j)
    
    def _calculate_proximity_reward(self, agent_pos):
        """
        Calculate a reward based on proximity to important objects.
        """
        if not self.proximity_reward:
            return 0.0
            
        proximity_reward = 0.0
        
        # Calculate distance to nearest key
        if self.key_positions and self.previous_keys == 0:  # Only if we haven't collected all keys
            min_key_distance = min([abs(agent_pos[0] - kp[0]) + abs(agent_pos[1] - kp[1]) 
                                 for kp in self.key_positions])
            # Reward for getting closer to a key
            if min_key_distance < self.previous_min_key_distance:
                proximity_reward += 0.01  # Small reward for moving closer to a key
            self.previous_min_key_distance = min_key_distance
        
        # Calculate distance to nearest door if we have a key
        if self.door_positions and self.previous_keys > 0:
            min_door_distance = min([abs(agent_pos[0] - dp[0]) + abs(agent_pos[1] - dp[1]) 
                                  for dp in self.door_positions])
            # Reward for getting closer to a door when we have a key
            if min_door_distance < self.previous_min_door_distance:
                proximity_reward += 0.02  # Slightly larger reward for moving closer to a door with a key
            self.previous_min_door_distance = min_door_distance
        
        # Calculate distance to goal if we have opened doors
        if self.goal_position and self.previous_doors > 0:
            goal_distance = abs(agent_pos[0] - self.goal_position[0]) + abs(agent_pos[1] - self.goal_position[1])
            # Reward for getting closer to the goal
            if goal_distance < self.previous_min_goal_distance:
                proximity_reward += 0.03  # Larger reward for moving closer to the goal after opening doors
            self.previous_min_goal_distance = goal_distance
        
        return proximity_reward
    
    def reset(self, seed=None):
        """
        Reset the environment and initialize frame stack with enhanced tracking.
        """
        obs, info = self.env.reset(seed=seed)
        
        # Reset tracking variables
        self.previous_keys = 0
        self.previous_doors = 0
        
        # Find object positions for proximity rewards
        self._find_object_positions(self.env.grid)
        
        # Initialize the frame stack with copies of the initial observation
        processed_obs = self._preprocess_observation(obs)
        
        # Each processed observation has 4 channels, so we'll repeat it for each frame
        # Debug: Print processed observation shape
        print(f"DEBUG: Processed obs shape: {processed_obs.shape}")
        
        # Create copies for frame stacking
        self.frames = deque(maxlen=self.frame_stack_size)
        for _ in range(self.frame_stack_size):
            # Deep copy to avoid reference issues
            frame_copy = processed_obs.copy()
            print(f"DEBUG: Frame copy shape: {frame_copy.shape}")
            self.frames.append(frame_copy)
            
        # Debug: Print frame stack sizes before concatenation
        print(f"DEBUG: Frame stack sizes: {[f.shape for f in self.frames]}")
        
        # Stack frames along the channel dimension
        try:
            stacked_obs = np.concatenate(list(self.frames), axis=2)
            print(f"DEBUG: Stacked obs shape: {stacked_obs.shape}")
        except Exception as e:
            print(f"DEBUG: Concatenation error: {e}")
            # Fallback to prevent crash during debugging
            stacked_obs = processed_obs
        
        return stacked_obs, info
    
    def step(self, action):
        """Take a step in the environment and update frame stack."""
        # Take action in the environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Process the observation
        processed_obs = self._preprocess_observation(obs)
        print(f"DEBUG: Step - Processed obs shape: {processed_obs.shape}")
        
        # Update the frame stack with the new observation
        self.frames.append(processed_obs)
        
        # Debug: Print frame stack sizes before concatenation
        print(f"DEBUG: Step - Frame stack sizes: {[f.shape for f in self.frames]}")
        
        # Stack frames into a single observation
        try:
            stacked_obs = np.concatenate(list(self.frames), axis=2)
            print(f"DEBUG: Step - Stacked obs shape: {stacked_obs.shape}")
        except Exception as e:
            print(f"DEBUG: Step - Concatenation error: {e}")
            # Fallback to prevent crash during debugging
            stacked_obs = processed_obs
        
        # Add proximity rewards to encourage key finding and door opening
        proximity_reward = 0.0
        exploration_bonus = 0.0
        
        if self.proximity_reward:
            # Add reward based on distance to nearest key or door
            if hasattr(self.env, 'agent_position'):
                agent_pos = self.env.agent_position
                proximity_reward = self._calculate_proximity_reward(agent_pos)
                reward += proximity_reward
            
            # Add reward for collecting keys and opening doors
            current_keys = info.get('keys_collected', 0)
            current_doors = info.get('doors_opened', 0)
            
            # Reward for new keys collected
            if current_keys > self.previous_keys:
                reward += 1.0
                self.previous_keys = current_keys
            
            # Reward for new doors opened
            if current_doors > self.previous_doors:
                reward += 2.0
                self.previous_doors = current_doors
            
            # Add exploration bonus for visiting new places
            if hasattr(self.env, 'visited_positions') and len(self.env.visited_positions) > 0:
                new_pos = tuple(self.env.agent_position)
                if new_pos not in self.env.visited_positions:
                    exploration_bonus = 0.01  # Small reward for exploring new areas
                    reward += exploration_bonus
        
        # Don't terminate episode when finding door without key
        if done and reward < 0.1:  # If episode ends without success
            if 'goal_reached' in info and not info['goal_reached']:
                done = False  # Don't end episode, continue learning
                truncated = False
        
        # Update info with enhanced details
        info["proximity_reward"] = proximity_reward
        info["exploration_bonus"] = exploration_bonus
        
        return stacked_obs, reward, done, truncated, info
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def get_coverage_stats(self):
        """Get coverage statistics from the environment."""
        return self.env.get_coverage_stats()
    
    def get_keys_collected(self):
        """Get the number of keys collected."""
        return sum(self.env.inventory)
    
    def close(self):
        """Close the environment."""
        return self.env.close() if hasattr(self.env, 'close') else None
