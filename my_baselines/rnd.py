"""
Random Network Distillation (RND) implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from .models import QNetwork, RNDPredictor, RNDTarget
from .utils import ReplayBuffer, preprocess_state, plot_learning_curve, epsilon_schedule

class RNDAgent:
    """
    Double DQN agent with Random Network Distillation for exploration.
    """
    
    def __init__(self, 
                 input_shape, 
                 num_actions, 
                 learning_rate=0.001,
                 rnd_learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_size=100000,
                 batch_size=64,
                 update_every=4,
                 target_update=1000,
                 rnd_output_dim=512,
                 intrinsic_weight=0.5,
                 intrinsic_reward_norm=True,
                 device=None):
        """
        Initialize the RND agent.
        
        Args:
            input_shape: Shape of the input observation (C, H, W)
            num_actions: Number of possible actions
            learning_rate: Learning rate for the Q-network optimizer
            rnd_learning_rate: Learning rate for the RND predictor optimizer
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            buffer_size: Size of the replay buffer
            batch_size: Number of samples to use for training
            update_every: How often to update the network
            target_update: How often to update the target network
            rnd_output_dim: Dimension of the RND feature embedding
            intrinsic_weight: Weight for intrinsic reward
            intrinsic_reward_norm: Whether to normalize intrinsic rewards
            device: Device to use for training (cpu/cuda)
        """
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update = target_update
        self.intrinsic_weight = intrinsic_weight
        self.intrinsic_reward_norm = intrinsic_reward_norm
        self.rnd_output_dim = rnd_output_dim
        self.steps = 0
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Q-networks (online and target)
        self.q_network = QNetwork(input_shape, num_actions).to(self.device)
        self.target_network = QNetwork(input_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is only used for inference
        
        # Initialize RND networks
        self.rnd_predictor = RNDPredictor(input_shape, rnd_output_dim).to(self.device)
        self.rnd_target = RNDTarget(input_shape, rnd_output_dim).to(self.device)
        
        # Initialize optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=rnd_learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize epsilon schedule
        self.epsilon_schedule = epsilon_schedule(epsilon_start, epsilon_end, epsilon_decay)
        self.epsilon = epsilon_start
        
        # Initialize intrinsic reward normalization
        self.intrinsic_reward_running_stats = {
            'mean': 0,
            'std': 1,
            'count': 0
        }
        
        # Training metrics
        self.q_losses = []
        self.rnd_losses = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.intrinsic_rewards = []
        self.total_intrinsic_reward = 0
        
        # Create directory for saving models
        self.checkpoint_dir = os.path.join("checkpoints", "rnd")
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
    
    def get_intrinsic_reward(self, state):
        """
        Calculate intrinsic reward based on RND prediction error.
        
        Args:
            state: Current state
            
        Returns:
            Intrinsic reward value
        """
        # Preprocess state
        state_tensor = preprocess_state(state).unsqueeze(0).to(self.device)
        
        # Get target and predictor outputs
        with torch.no_grad():
            target_features = self.rnd_target(state_tensor)
            predictor_features = self.rnd_predictor(state_tensor)
            
            # Calculate prediction error (MSE)
            error = ((target_features - predictor_features) ** 2).mean().item()
            
        # Update running statistics for normalization
        if self.intrinsic_reward_norm:
            self.update_intrinsic_reward_stats(error)
            # Normalize the reward
            if self.intrinsic_reward_running_stats['count'] > 1:
                std = max(1e-8, self.intrinsic_reward_running_stats['std'])
                error = (error - self.intrinsic_reward_running_stats['mean']) / std
            
        return error
    
    def update_intrinsic_reward_stats(self, reward):
        """
        Update running statistics for intrinsic reward normalization.
        
        Args:
            reward: New intrinsic reward value
        """
        n = self.intrinsic_reward_running_stats['count']
        mean = self.intrinsic_reward_running_stats['mean']
        var = self.intrinsic_reward_running_stats['std'] ** 2
        
        # Welford's online algorithm for variance
        n += 1
        delta = reward - mean
        mean += delta / n
        delta2 = reward - mean
        var = (var * (n - 1) + delta * delta2) / n
        
        self.intrinsic_reward_running_stats['count'] = n
        self.intrinsic_reward_running_stats['mean'] = mean
        self.intrinsic_reward_running_stats['std'] = np.sqrt(var)
    
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
        # Calculate intrinsic reward
        intrinsic_reward = self.get_intrinsic_reward(next_state)  # Use next_state for curiosity
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
        """Update the Q-network and RND predictor using a batch of experiences."""
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update RND predictor
        self._update_rnd_predictor(states)
        
        # Update Q-network
        self._update_q_network(states, actions, rewards, next_states, dones)
    
    def _update_rnd_predictor(self, states):
        """
        Update the RND predictor network to predict the output of the RND target.
        
        Args:
            states: Batch of states
        """
        # Get target and predictor outputs
        with torch.no_grad():
            target_features = self.rnd_target(states)
        predictor_features = self.rnd_predictor(states)
        
        # Calculate prediction error (MSE)
        rnd_loss = nn.MSELoss()(predictor_features, target_features)
        
        # Update predictor
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        
        # Track loss
        self.rnd_losses.append(rnd_loss.item())
    
    def _update_q_network(self, states, actions, rewards, next_states, dones):
        """
        Update the Q-network using DDQN.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        """
        # Get current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next actions using online network (for DDQN)
        next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
        
        # Get next Q-values using target network (for DDQN)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        q_loss = nn.MSELoss()(q_values, targets)
        
        # Update network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Track loss
        self.q_losses.append(q_loss.item())
    
    def save_checkpoint(self, filename=None):
        """
        Save the agent's state.
        
        Args:
            filename: Name of the checkpoint file
        """
        if filename is None:
            filename = f"rnd_{time.strftime('%Y%m%d_%H%M%S')}.pt"
            
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'rnd_predictor_state_dict': self.rnd_predictor.state_dict(),
            'rnd_target_state_dict': self.rnd_target.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'rnd_optimizer_state_dict': self.rnd_optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
            'q_losses': self.q_losses,
            'rnd_losses': self.rnd_losses,
            'episode_rewards': self.episode_rewards,
            'intrinsic_rewards': self.intrinsic_rewards,
            'intrinsic_reward_running_stats': self.intrinsic_reward_running_stats
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
        self.rnd_predictor.load_state_dict(checkpoint['rnd_predictor_state_dict'])
        self.rnd_target.load_state_dict(checkpoint['rnd_target_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
        self.q_losses = checkpoint['q_losses']
        self.rnd_losses = checkpoint['rnd_losses']
        self.episode_rewards = checkpoint['episode_rewards']
        if 'intrinsic_rewards' in checkpoint:
            self.intrinsic_rewards = checkpoint['intrinsic_rewards']
        if 'intrinsic_reward_running_stats' in checkpoint:
            self.intrinsic_reward_running_stats = checkpoint['intrinsic_reward_running_stats']
        
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
        Plot the Q-network and RND losses over training steps.
        
        Args:
            window_size: Window size for smoothing
            filename: If provided, save the figure to this file
        """
        plt.figure(figsize=(12, 8))
        
        # Plot Q-network losses
        plt.subplot(2, 1, 1)
        if len(self.q_losses) > 0:
            plt.plot(self.q_losses, alpha=0.3, color='red', label='Q Loss')
            
            # Plot smoothed losses
            if len(self.q_losses) >= window_size:
                smooth_losses = []
                for i in range(len(self.q_losses) - window_size + 1):
                    smooth_losses.append(np.mean(self.q_losses[i:i+window_size]))
                plt.plot(range(window_size-1, len(self.q_losses)), smooth_losses, 
                         color='red', label=f'Smoothed (window={window_size})')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Q-Network Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot RND losses
        plt.subplot(2, 1, 2)
        if len(self.rnd_losses) > 0:
            plt.plot(self.rnd_losses, alpha=0.3, color='blue', label='RND Loss')
            
            # Plot smoothed losses
            if len(self.rnd_losses) >= window_size:
                smooth_losses = []
                for i in range(len(self.rnd_losses) - window_size + 1):
                    smooth_losses.append(np.mean(self.rnd_losses[i:i+window_size]))
                plt.plot(range(window_size-1, len(self.rnd_losses)), smooth_losses, 
                         color='blue', label=f'Smoothed (window={window_size})')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('RND Predictor Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        
        plt.show()
        
    def plot_feature_maps(self, state, filename=None):
        """
        Visualize the RND target and predictor embeddings for a state.
        
        Args:
            state: State to visualize embeddings for
            filename: If provided, save the figure to this file
        """
        # Preprocess state
        state_tensor = preprocess_state(state).unsqueeze(0).to(self.device)
        
        # Get target and predictor outputs
        with torch.no_grad():
            target_features = self.rnd_target(state_tensor).cpu().numpy()[0]
            predictor_features = self.rnd_predictor(state_tensor).cpu().numpy()[0]
            
        # Plot feature maps
        plt.figure(figsize=(12, 6))
        
        # Target features
        plt.subplot(2, 1, 1)
        plt.bar(range(min(50, len(target_features))), target_features[:50])
        plt.title('RND Target Network Features (first 50)')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        
        # Predictor features
        plt.subplot(2, 1, 2)
        plt.bar(range(min(50, len(predictor_features))), predictor_features[:50])
        plt.title('RND Predictor Network Features (first 50)')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            
        plt.show()
