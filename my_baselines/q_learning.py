"""
Standard Q-Learning implementation with neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from .models import QNetwork
from .utils import ReplayBuffer, preprocess_state, plot_learning_curve, epsilon_schedule

class QLearningAgent:
    """
    Standard Q-Learning agent with neural networks.
    """
    
    def __init__(self, 
                 input_shape, 
                 num_actions, 
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_size=100000,
                 batch_size=64,
                 update_every=4,
                 device=None):
        """
        Initialize the Q-Learning agent.
        
        Args:
            input_shape: Shape of the input observation (C, H, W)
            num_actions: Number of possible actions
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            buffer_size: Size of the replay buffer
            batch_size: Number of samples to use for training
            update_every: How often to update the network
            device: Device to use for training (cpu/cuda)
        """
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.steps = 0
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Q-network
        self.q_network = QNetwork(input_shape, num_actions).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize epsilon schedule
        self.epsilon_schedule = epsilon_schedule(epsilon_start, epsilon_end, epsilon_decay)
        self.epsilon = epsilon_start
        
        # Training metrics
        self.losses = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Create directory for saving models
        self.checkpoint_dir = os.path.join("checkpoints", "q_learning")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def select_action(self, state, epsilon=None):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate (if None, use current epsilon)
            
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        # With probability epsilon, select a random action
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        
        # Otherwise, select the action with highest Q-value
        with torch.no_grad():
            state_tensor = preprocess_state(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def step(self, state, action, reward, next_state, done, episode):
        """
        Take a step in the environment and update the agent.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            episode: Current episode number
        """
        # Add experience to replay buffer
        self.memory.add(preprocess_state(state).cpu().numpy(),
                       action, reward, preprocess_state(next_state).cpu().numpy(), done)
        
        # Track rewards for the current episode
        self.current_episode_reward += reward
        
        # If episode is done, reset episode reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            # Update epsilon based on episode
            self.epsilon = self.epsilon_schedule(episode)
        
        # Increment step counter
        self.steps += 1
        
        # Only update every update_every steps and if we have enough samples
        if self.steps % self.update_every == 0 and len(self.memory) > self.batch_size:
            self._learn()
    
    def _learn(self):
        """Update the Q-network using a batch of experiences."""
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q-values for current states and actions
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(q_values, targets)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track loss
        self.losses.append(loss.item())
    
    def save_checkpoint(self, filename=None):
        """
        Save the agent's state.
        
        Args:
            filename: Name of the checkpoint file
        """
        if filename is None:
            filename = f"q_learning_{time.strftime('%Y%m%d_%H%M%S')}.pt"
            
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
            'losses': self.losses,
            'episode_rewards': self.episode_rewards
        }, filepath)
        
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load the agent's state from a checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        self.episode_rewards = checkpoint['episode_rewards']
        
        print(f"Loaded checkpoint from {filepath}")
    
    def plot_rewards(self, filename=None):
        """
        Plot the rewards over episodes.
        
        Args:
            filename: If provided, save the figure to this file
        """
        plot_learning_curve(self.episode_rewards, filename=filename)
    
    def plot_losses(self, window_size=100, filename=None):
        """
        Plot the losses over training steps.
        
        Args:
            window_size: Window size for smoothing
            filename: If provided, save the figure to this file
        """
        plt.figure(figsize=(10, 6))
        
        # Plot losses
        if len(self.losses) > 0:
            plt.plot(self.losses, alpha=0.3, color='red', label='Loss')
            
            # Plot smoothed losses
            if len(self.losses) >= window_size:
                smooth_losses = []
                for i in range(len(self.losses) - window_size + 1):
                    smooth_losses.append(np.mean(self.losses[i:i+window_size]))
                plt.plot(range(window_size-1, len(self.losses)), smooth_losses, 
                         color='red', label=f'Smoothed (window={window_size})')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        if filename:
            plt.savefig(filename)
        
        plt.show()
