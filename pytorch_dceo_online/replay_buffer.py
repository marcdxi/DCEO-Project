"""
Replay buffer implementations for the fully online Rainbow DCEO agent.
"""

import numpy as np
import torch
import random
from collections import namedtuple

# Define transition tuple structure
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for training.
    
    Based on the paper: "Prioritized Experience Replay" (Schaul et al., 2016).
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta_start = beta_start  # Start value of beta for importance-sampling
        self.beta_frames = beta_frames  # Number of frames over which to anneal beta
        self.frame = 1  # Current frame count
        
        self.buffer = []  # Experience buffer
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0  # Current position in circular buffer
    
    def push(self, state, action, reward, next_state, done, priority=None):
        """Store new experience with priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = Transition(state, action, reward, next_state, done)
        
        # Use max priority for new experiences
        self.priorities[self.position] = priority if priority is not None else max_priority
        
        # Move position pointer
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, device):
        # Update beta value based on current frame
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample experiences based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability
        
        # Memory-optimized batch preparation for PyTorch
        # Process samples in smaller chunks to avoid large memory allocations
        try:
            # Try the standard approach first
            states = torch.FloatTensor(np.array([s.state for s in samples])).to(device)
            next_states = torch.FloatTensor(np.array([s.next_state for s in samples])).to(device)
        except (np.core._exceptions._ArrayMemoryError, MemoryError):
            # If we hit a memory error, use a more memory-efficient approach
            # Process in smaller batches
            chunk_size = max(1, batch_size // 4)  # Process 1/4 of the batch at a time
            
            # Process states
            states_list = []
            for i in range(0, batch_size, chunk_size):
                chunk_samples = samples[i:i+chunk_size]
                chunk_states = torch.FloatTensor(np.array([s.state for s in chunk_samples]))
                states_list.append(chunk_states)
            states = torch.cat(states_list).to(device)
            
            # Process next_states
            next_states_list = []
            for i in range(0, batch_size, chunk_size):
                chunk_samples = samples[i:i+chunk_size]
                chunk_next_states = torch.FloatTensor(np.array([s.next_state for s in chunk_samples]))
                next_states_list.append(chunk_next_states)
            next_states = torch.cat(next_states_list).to(device)
            
            # Release any temporary arrays to help with memory
            import gc
            gc.collect()
        
        # These are smaller arrays, so process them directly
        actions = torch.tensor([s.action for s in samples], dtype=torch.long).to(device)
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float).to(device)
        dones = torch.tensor([s.done for s in samples], dtype=torch.float).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        # Increment frame counter
        self.frame += batch_size
        
        return (states, actions, rewards, next_states, dones, weights, indices)
    
    def update_priorities(self, indices, priorities):
        """Update priorities after learning."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)
