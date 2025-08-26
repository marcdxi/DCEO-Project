"""
Utility functions and classes for baseline RL algorithms.
"""

import numpy as np
import torch
import random
from collections import deque, defaultdict
import matplotlib.pyplot as plt

class ReplayBuffer:
    """
    Experience replay buffer to store transitions for off-policy learning.
    """
    
    def __init__(self, capacity):
        """
        Initialize replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors - carefully handling potentially different state shapes
        # For states and next_states, directly stack the arrays (they should already be preprocessed)
        states_tensor = torch.stack([torch.FloatTensor(s) for s in states])
        next_states_tensor = torch.stack([torch.FloatTensor(s) for s in next_states])
        
        # Other elements can be converted normally
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        dones = torch.FloatTensor(np.array(dones))
        
        return states_tensor, actions, rewards, next_states_tensor, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class StateCountTable:
    """
    Table to track state visitation counts for count-based exploration.
    Uses a discretization strategy for continuous state spaces.
    """
    
    def __init__(self, n_bins=8):
        """
        Initialize the count table.
        
        Args:
            n_bins: Number of bins for discretization
        """
        self.counts = defaultdict(int)
        self.n_bins = n_bins
        self.state_min = None
        self.state_max = None
        self.total_count = 0
    
    def update_ranges(self, state):
        """Update min/max ranges for state normalization."""
        if self.state_min is None or self.state_max is None:
            self.state_min = np.array(state)
            self.state_max = np.array(state)
        else:
            self.state_min = np.minimum(self.state_min, state)
            self.state_max = np.maximum(self.state_max, state)
    
    def discretize(self, state):
        """
        Discretize a continuous state into bins.
        
        Args:
            state: Continuous state array or tuple containing state information
            
        Returns:
            Tuple or hash representing the discretized state
        """
        # Handle complex maze states (might be tuples or contain additional info)
        try:
            # For tensor/array states from image-based environments
            if hasattr(state, 'shape') and len(state.shape) >= 3:
                # Already a tensor/array with channel dimension
                state_arr = state
            elif isinstance(state, (list, tuple)) and hasattr(state[0], 'shape'):
                # First element is the observation tensor/array
                state_arr = state[0]
            else:
                # Try standard conversion
                state_arr = np.array(state)
            
            # Update min/max ranges
            self.update_ranges(state_arr)
            
            # Handle image-based states
            if len(state_arr.shape) >= 2:  # For image states (maze environments)
                # Reduce dimensionality by averaging across channels if needed
                if len(state_arr.shape) > 2:
                    # Average across channel dimension for consistency
                    state_arr = np.mean(state_arr, axis=0) if state_arr.shape[0] <= 3 else np.mean(state_arr, axis=-1)
                
                # Resize and discretize the image
                from skimage.transform import resize
                small_state = resize(state_arr, (self.n_bins, self.n_bins), anti_aliasing=True)
                return hash(small_state.tobytes())
            
            # For vector states (e.g., Mountain Car)
            epsilon = 1e-10  # Avoid division by zero
            normalized = (state_arr - self.state_min) / (self.state_max - self.state_min + epsilon)
            discretized = tuple(min(self.n_bins - 1, int(x * self.n_bins)) for x in normalized)
            return discretized
            
        except (ValueError, TypeError) as e:
            # Fallback for complex states that can't be converted directly
            # Create a string representation and hash it
            return hash(str(state))
    
    def update(self, state):
        """
        Update the count for a state.
        
        Args:
            state: State to update
            
        Returns:
            New count for the state
        """
        discrete_state = self.discretize(state)
        self.counts[discrete_state] += 1
        self.total_count += 1
        return self.counts[discrete_state]
    
    def get_count(self, state):
        """
        Get the current count for a state.
        
        Args:
            state: State to get count for
            
        Returns:
            Current count for the state
        """
        discrete_state = self.discretize(state)
        return self.counts.get(discrete_state, 0)
    
    def get_bonus(self, state, beta=0.2):
        """
        Calculate the exploration bonus based on state count.
        
        Formula: beta / sqrt(count(s))
        
        Args:
            state: State to get bonus for
            beta: Bonus scaling factor
            
        Returns:
            Exploration bonus value
        """
        count = max(1, self.get_count(state))
        return beta / np.sqrt(count)


def preprocess_state(state, target_size=(12, 12)):
    """
    Preprocess the state for model input.
    
    Args:
        state: Raw state from environment
        target_size: Target size for the maze (height, width)
        
    Returns:
        Preprocessed state as torch tensor with consistent dimensions
    """
    # State preprocessing without debug prints
    
    # Handle tuple states (from maze environments reset method)
    if isinstance(state, tuple):
        # For tuple observations, extract the main observation
        if len(state) > 0:
            state = state[0]  # Extract observation
        else:
            # Create a default empty state if tuple is empty
            state = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32) 
    
    # Ensure we have a numpy array
    if not isinstance(state, np.ndarray) and not isinstance(state, torch.Tensor):
        try:
            state = np.array(state, dtype=np.float32)
        except:
            # If conversion fails, create an empty state
            state = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
    
    # Convert numpy array to torch tensor
    if isinstance(state, np.ndarray):
        # Check the shape for image data (maze environment output)
        if len(state.shape) == 3:  # Image shape (H, W, C) or (C, H, W)
            # Check if state is already in (C, H, W) format
            if state.shape[0] in [3, 5] and state.shape[1] == state.shape[2]:
                # Already in (C, H, W) format
                state_tensor = torch.FloatTensor(state)
                return state_tensor
                
            # For maze observations, ensure consistent dimensions
            if state.shape[:2] != target_size:
                num_channels = state.shape[2]  # Could be 3 for ComplexMazeEnv or 5 for KeyDoorMazeEnv
                new_state = np.zeros((target_size[0], target_size[1], num_channels), dtype=np.float32)
                # Copy as much as possible from the original state
                h = min(state.shape[0], target_size[0])
                w = min(state.shape[1], target_size[1])
                new_state[:h, :w, :] = state[:h, :w, :]
                state = new_state
                
            # Convert from (H, W, C) to (C, H, W) for PyTorch
            return torch.FloatTensor(state).permute(2, 0, 1)
        else:
            # For other array shapes
            return torch.FloatTensor(state)
    elif isinstance(state, torch.Tensor):
        # If already a tensor, make sure it's in (C, H, W) format if it's an image
        if len(state.shape) == 3:
            # Check if in (H, W, C) format and needs to be converted
            if state.shape[0] > 10 and state.shape[2] in [3, 5]:  # Likely (H, W, C) for maze (assuming maze dimension > 10)
                # Convert from (H, W, C) to (C, H, W)
                return state.permute(2, 0, 1).float()
            # If neither channel dim is right, reshape based on total size
            elif state.shape[0] not in [3, 5] and state.shape[2] not in [3, 5]:
                # Try to infer the correct shape
                if state.shape[0] == state.shape[1]:  # Square image, assume (H, W, C) format
                    total_elements = state.numel()
                    if total_elements % (state.shape[0] * state.shape[1]) == 0:
                        inferred_channels = total_elements // (state.shape[0] * state.shape[1])
                        # Reshape to (H, W, C) then permute to (C, H, W)
                        return state.reshape(state.shape[0], state.shape[1], inferred_channels).permute(2, 0, 1).float()
        # Default: return as is but ensure float type
        return state.float()
    
    # If all else fails, return a default state
    return torch.zeros((3, target_size[0], target_size[1]), dtype=torch.float32)


def plot_learning_curve(rewards, window_size=10, filename=None):
    """
    Plot the learning curve (rewards over episodes).
    
    Args:
        rewards: List of episode rewards
        window_size: Window size for smoothing
        filename: If provided, save the figure to this file
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Plot smoothed rewards
    if len(rewards) >= window_size:
        smooth_rewards = []
        for i in range(len(rewards) - window_size + 1):
            smooth_rewards.append(np.mean(rewards[i:i+window_size]))
        plt.plot(range(window_size-1, len(rewards)), smooth_rewards, color='blue', label=f'Smoothed (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    
    if filename:
        plt.savefig(filename)
    
    plt.show()


def epsilon_schedule(start=1.0, end=0.01, decay=0.995):
    """
    Create an epsilon decay schedule for exploration.
    
    Args:
        start: Starting epsilon value
        end: Minimum epsilon value
        decay: Decay rate
        
    Returns:
        Function that takes the episode number and returns epsilon
    """
    def schedule(episode):
        return max(end, start * (decay ** episode))
    
    return schedule
