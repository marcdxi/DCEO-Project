"""
Key-Door Maze Wrapper for Rainbow DCEO Agent.

This wrapper adapts the Key-Door Maze environment to work with the
Rainbow DCEO agent by handling preprocessing and frame stacking.
"""

import gym
import numpy as np
from collections import deque
import cv2

class DCEOKeyMazeWrapper(gym.Wrapper):
    """
    Wrapper that adapts Key-Door Maze environment to work with Rainbow DCEO agents.
    
    It handles:
    1. Frame stacking (combining multiple frames for temporal information)
    2. Observation preprocessing (resizing, channel transformation)
    3. Reward shaping (adding proximity rewards, exploration bonuses)
    """
    
    def __init__(self, env, frame_stack=4, resize_shape=(84, 84), proximity_reward=True):
        """Initialize the wrapper.
        
        Args:
            env: The Key-Door Maze environment to wrap
            frame_stack: Number of frames to stack
            resize_shape: Target shape for observations (H, W)
            proximity_reward: Whether to add proximity-based rewards
        """
        super(DCEOKeyMazeWrapper, self).__init__(env)
        
        self.env = env
        self.frame_stack_size = frame_stack
        self.resize_shape = resize_shape
        self.proximity_reward = proximity_reward
        
        # Create frame buffer for frame stacking
        self.frames = deque(maxlen=frame_stack)
        
        # Initialize variables for reward shaping
        self.previous_keys = 0
        self.previous_doors = 0
        
        # Initialize tracking for proximity rewards
        self.previous_min_key_distance = float('inf')
        self.previous_min_door_distance = float('inf')
        self.previous_min_goal_distance = float('inf')
        
        # Update observation space to account for frame stacking in CHW format (PyTorch style)
        h, w = resize_shape
        channels = 4  # We'll use 4 channels in our processed observations
        stacked_channels = channels * frame_stack
        
        # Note: Using CHW format (PyTorch style) instead of HWC
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(stacked_channels, h, w),  # CHW instead of HWC
            dtype=np.float32
        )
        
        # Keep action space the same
        self.action_space = env.action_space
        
    def _preprocess_observation(self, obs):
        """
        Convert multi-channel observation to 4 essential channels and resize.
        Returns observation in NCHW format for PyTorch.
        
        Channels:
        0: Agent position
        1: Keys
        2: Doors
        3: Walls and goal
        """
        # Extract and resize individual channels
        agent_channel = cv2.resize(obs[:, :, 0], self.resize_shape, interpolation=cv2.INTER_AREA)
        walls_channel = cv2.resize(obs[:, :, 1], self.resize_shape, interpolation=cv2.INTER_AREA)
        goal_channel = cv2.resize(obs[:, :, 2], self.resize_shape, interpolation=cv2.INTER_AREA)
        keys_channel = cv2.resize(obs[:, :, 3], self.resize_shape, interpolation=cv2.INTER_AREA)
        doors_channel = cv2.resize(obs[:, :, 4], self.resize_shape, interpolation=cv2.INTER_AREA)
        
        # Create a 4-channel observation in CHW format (PyTorch style)
        # Shape: (4, H, W) instead of (H, W, 4)
        processed = np.zeros((4, self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)
        
        # Assign channels
        processed[0] = agent_channel  # Agent position
        processed[1] = keys_channel   # Keys
        processed[2] = doors_channel  # Doors
        processed[3] = walls_channel + goal_channel  # Walls and goal combined
        
        return processed
        
    def _calculate_proximity_reward(self, agent_pos):
        """Calculate reward based on proximity to keys, doors, and goal."""
        proximity_reward = 0.0
        
        # Calculate distance to nearest key if any keys are uncollected
        min_key_distance = float('inf')
        for i, key_pos in enumerate(self.env.key_positions):
            if not self.env.inventory[i]:
                distance = abs(agent_pos[0] - key_pos[0]) + abs(agent_pos[1] - key_pos[1])
                min_key_distance = min(min_key_distance, distance)
        
        # Add reward for getting closer to keys
        if min_key_distance < float('inf') and min_key_distance < self.previous_min_key_distance:
            proximity_reward += 0.02
        
        # Update previous distance
        if min_key_distance < float('inf'):
            self.previous_min_key_distance = min_key_distance
        
        # Calculate distance to nearest door if we have any keys
        min_door_distance = float('inf')
        if sum(self.env.inventory) > 0:
            for i, door_pos in enumerate(self.env.door_positions):
                if self.env.inventory[i] and not self.env.door_status[i]:
                    distance = abs(agent_pos[0] - door_pos[0]) + abs(agent_pos[1] - door_pos[1])
                    min_door_distance = min(min_door_distance, distance)
            
            # Add reward for getting closer to doors
            if min_door_distance < float('inf') and min_door_distance < self.previous_min_door_distance:
                proximity_reward += 0.02
        
        # Update previous distance
        if min_door_distance < float('inf'):
            self.previous_min_door_distance = min_door_distance
        
        # Calculate distance to goal if all doors are open
        if self.env.door_status.count(True) == len(self.env.door_positions):
            goal_distance = abs(agent_pos[0] - self.env.goal_position[0]) + abs(agent_pos[1] - self.env.goal_position[1])
            
            # Add reward for getting closer to goal
            if goal_distance < self.previous_min_goal_distance:
                proximity_reward += 0.03
            
            self.previous_min_goal_distance = goal_distance
        
        return proximity_reward
    
    def reset(self, seed=None):
        """Reset the environment and initialize the frame stack."""
        # Reset the environment
        obs, info = self.env.reset(seed=seed)
        
        # Reset tracking variables
        self.previous_keys = 0
        self.previous_doors = 0
        self.previous_min_key_distance = float('inf')
        self.previous_min_door_distance = float('inf')
        self.previous_min_goal_distance = float('inf')
        
        # Process the observation - now in CHW format (C=4)
        processed_obs = self._preprocess_observation(obs)
        
        # Initialize frame stack with copies of the initial observation
        self.frames = deque(maxlen=self.frame_stack_size)
        for _ in range(self.frame_stack_size):
            self.frames.append(processed_obs.copy())
        
        # Stack frames along the channel dimension (axis=0 for CHW format)
        # Each frame is (4, H, W), so stacked will be (4*frame_stack_size, H, W)
        stacked_obs = np.concatenate(list(self.frames), axis=0)
        
        return stacked_obs, info
    
    def step(self, action):
        """Take a step in the environment and update the frame stack."""
        # Take action in the environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Process the observation - now in CHW format (C=4)
        processed_obs = self._preprocess_observation(obs)
        
        # Update the frame stack
        self.frames.append(processed_obs)
        
        # Stack frames along the channel dimension (axis=0 for CHW format)
        # Each frame is (4, H, W), so stacked will be (4*frame_stack_size, H, W)
        stacked_obs = np.concatenate(list(self.frames), axis=0)
        
        # Add proximity rewards and exploration bonuses
        proximity_reward = 0.0
        exploration_bonus = 0.0
        
        if self.proximity_reward:
            # Calculate proximity reward
            agent_pos = self.env.agent_position
            proximity_reward = self._calculate_proximity_reward(agent_pos)
            reward += proximity_reward
            
            # Add reward for collecting keys
            current_keys = sum(self.env.inventory)
            if current_keys > self.previous_keys:
                reward += 0.5
                self.previous_keys = current_keys
                
            # Add reward for opening doors
            current_doors = sum(self.env.door_status)
            if current_doors > self.previous_doors:
                reward += 1.0
                self.previous_doors = current_doors
            
            # Add exploration bonus for visiting new cells
            new_pos = self.env.agent_position
            visited_count = len(self.env.visited_positions)
            if visited_count > 0:
                if new_pos not in self.env.visited_positions:
                    exploration_bonus = 0.01
                    reward += exploration_bonus
        
        # Store exploration bonuses and proximity rewards in info
        info['proximity_reward'] = proximity_reward
        info['exploration_bonus'] = exploration_bonus
        
        return stacked_obs, reward, done, truncated, info
