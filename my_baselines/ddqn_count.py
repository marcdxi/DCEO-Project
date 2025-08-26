"""
Double DQN with count-based exploration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from .models import QNetwork
from .utils import ReplayBuffer, StateCountTable, preprocess_state, plot_learning_curve, epsilon_schedule

class DDQNCountAgent:
    """
    Double DQN agent with count-based exploration.
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
                 target_update=1000,
                 count_beta=0.2,
                 intrinsic_weight=0.5,
                 device=None):
        """
        Initialize the DDQN agent with count-based exploration.
        
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
            target_update: How often to update the target network
            count_beta: Beta parameter for count-based bonus
            intrinsic_weight: Weight for intrinsic reward
            device: Device to use for training (cpu/cuda)
        """
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update = target_update
        self.count_beta = count_beta
        self.intrinsic_weight = intrinsic_weight
        
        # Initialize step counter
        self.steps = 0
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Q-networks (online and target)
        self.q_network = QNetwork(input_shape, num_actions).to(self.device)
        self.target_network = QNetwork(input_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize state count table for exploration bonus
        self.state_counter = StateCountTable()
        
        # Initialize epsilon schedule
        self.epsilon_schedule = epsilon_schedule(epsilon_start, epsilon_end, epsilon_decay)
        self.epsilon = epsilon_start
        
        # Training metrics
        self.losses = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.intrinsic_rewards = []
        self.total_intrinsic_reward = 0
        
        # Create directory for saving models
        self.checkpoint_dir = os.path.join("checkpoints", "ddqn_count")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def select_action(self, state, epsilon=None):
        """
        Select action according to an epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate (if None, use current epsilon)
            
        Returns:
            Selected action
        """
        # Force high exploration for the first 1000 steps
        if self.steps < 1000:
            forced_epsilon = max(0.7, self.epsilon)  # At least 70% random actions to start
        else:
            forced_epsilon = self.epsilon
            
        if epsilon is None:
            epsilon = forced_epsilon
        
        # No debug prints for action selection
            
        # With probability epsilon, select a random action
        if np.random.random() < epsilon:
            action = np.random.randint(self.num_actions)
            return action
        
        # Otherwise, select the action with highest Q-value
        with torch.no_grad():
            state_tensor = preprocess_state(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
            return action
    
    def get_intrinsic_reward(self, state):
        """
        Calculate intrinsic reward based on state visit counts.
        
        Args:
            state: Current state
            
        Returns:
            Intrinsic reward value
        """
        # Get exploration bonus based on state counts
        intrinsic_reward = self.state_counter.get_bonus(state, self.count_beta)
        return intrinsic_reward
    
    def step(self, state, action, reward, next_state, done, episode):
        """
        Take a step in the environment and update the agent.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received (extrinsic)
            next_state: Next state
            done: Whether the episode is done
            episode: Current episode number
        """
        # Increment step count
        self.steps += 1
        
        # Update state count table
        self.state_counter.update(state)
        
        # Calculate intrinsic reward
        intrinsic_reward = self.get_intrinsic_reward(state)
        self.total_intrinsic_reward += intrinsic_reward
        
        # Combine extrinsic and intrinsic rewards
        combined_reward = reward + self.intrinsic_weight * intrinsic_reward
        
        # Add experience to replay buffer
        self.memory.add(preprocess_state(state).cpu().numpy(),
                       action, combined_reward, preprocess_state(next_state).cpu().numpy(), done)
        
        # Track rewards for the current episode
        self.current_episode_reward += reward  # Only track extrinsic rewards
        
        # If episode is done, reset episode reward and track intrinsic rewards
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.intrinsic_rewards.append(self.total_intrinsic_reward)
            self.current_episode_reward = 0
            self.total_intrinsic_reward = 0
            # Update epsilon based on episode
            self.epsilon = self.epsilon_schedule(episode)
        
        # Increment step counter
        self.steps += 1
        
        # Only update every update_every steps and if we have enough samples
        if self.steps % self.update_every == 0 and len(self.memory) > self.batch_size:
            self._learn()
            
        # Update target network periodically
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
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
        
        # Get current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next actions using online network (for DDQN)
        next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
        
        # Get next Q-values using target network (for DDQN)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
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
            filename = f"ddqn_count_{time.strftime('%Y%m%d_%H%M%S')}.pt"
            
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
            'losses': self.losses,
            'episode_rewards': self.episode_rewards,
            'intrinsic_rewards': self.intrinsic_rewards
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
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        self.episode_rewards = checkpoint['episode_rewards']
        if 'intrinsic_rewards' in checkpoint:
            self.intrinsic_rewards = checkpoint['intrinsic_rewards']
        
        print(f"Loaded checkpoint from {filepath}")
    
    def plot_rewards(self, filename=None):
        """
        Plot the rewards over episodes.
        
        Args:
            filename: If provided, save the figure to this file
        """
        plot_learning_curve(self.episode_rewards, filename=filename)
        
    def plot_intrinsic_rewards(self, window_size=10, filename=None):
        """
        Plot the intrinsic rewards over episodes.
        
        Args:
            window_size: Window size for smoothing
            filename: If provided, save the figure to this file
        """
        plt.figure(figsize=(10, 6))
        
        # Plot raw intrinsic rewards
        plt.plot(self.intrinsic_rewards, alpha=0.3, color='green', label='Raw Intrinsic Rewards')
        
        # Plot smoothed intrinsic rewards
        if len(self.intrinsic_rewards) >= window_size:
            smooth_rewards = []
            for i in range(len(self.intrinsic_rewards) - window_size + 1):
                smooth_rewards.append(np.mean(self.intrinsic_rewards[i:i+window_size]))
            plt.plot(range(window_size-1, len(self.intrinsic_rewards)), smooth_rewards, 
                     color='green', label=f'Smoothed (window={window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Intrinsic Reward')
        plt.title('Intrinsic Rewards per Episode')
        plt.legend()
        plt.grid(True)
        
        if filename:
            plt.savefig(filename)
        
        plt.show()
    
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
    
    def plot_state_counts(self, env, grid_size=None, filename=None):
        """
        Plot a heatmap of state visitation counts.
        Only works for environments with 2D state spaces like mazes.
        
        Args:
            env: Environment to visualize
            grid_size: Size of the grid (if None, use env.maze_size)
            filename: If provided, save the figure to this file
        """
        if not hasattr(env, 'maze_size'):
            print("Environment does not support state count visualization")
            return
            
        if grid_size is None:
            grid_size = env.maze_size
            
        # Create count matrix
        count_matrix = np.zeros((grid_size, grid_size))
        
        # Fill in counts
        for x in range(grid_size):
            for y in range(grid_size):
                # Create a fake state with the agent at this position
                fake_state = np.zeros((grid_size, grid_size, 3))  # RGB channels
                fake_state[y, x, 0] = 1.0  # Agent in red channel
                count = self.state_counter.get_count(fake_state)
                count_matrix[y, x] = count
                
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(count_matrix, cmap='viridis')
        plt.colorbar(label='Visit Count')
        plt.title('State Visitation Counts')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            
        plt.show()
