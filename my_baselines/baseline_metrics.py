"""
Baseline Metrics - Tracking metrics for baseline reinforcement learning agents
Based on metrics from Key Maze DCEO implementation but adapted for simpler baselines.

This module implements the following metrics categories:
- Performance Metrics (rewards, success rate, episode lengths)
- Exploration Metrics (state coverage, visitation heatmaps)
"""

import os
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import torch

class BaselineMetricsTracker:
    """
    Metrics tracker for baseline reinforcement learning agents.
    Implements core metrics from the DCEO paper methodology.
    """
    
    def __init__(self, env, agent, config=None, base_dir="baseline_results"):
        """Initialize the metrics tracker.
        
        Args:
            env: The environment (maze environment)
            agent: The baseline agent (Q-Learning, DDQN, RND)
            config: Configuration dictionary with training parameters
            base_dir: Base directory for results
        """
        self.env = env
        self.agent = agent
        self.config = config or {}
        self.agent_name = getattr(agent, 'name', type(agent).__name__)
        
        # Create timestamp for this training run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get environment info from passed arguments rather than class name
        self.env_name = getattr(env, 'name', env.__class__.__name__.lower())
        self.maze_size = getattr(env, 'maze_size', 15)
        self.fixed_layout = getattr(env, 'use_fixed_layout', False)
        
        # Create results directory for this run
        model_env_dir = f"{self.agent_name}_{self.env_name}_maze{self.maze_size}{'_fixed' if self.fixed_layout else ''}"
        self.results_dir = os.path.join(base_dir, model_env_dir, self.timestamp)
        
        # Create subdirectories for different metric categories
        self.dirs = {
            'performance': os.path.join(self.results_dir, "performance_metrics"),
            'exploration': os.path.join(self.results_dir, "exploration_metrics"),
            'plots': os.path.join(self.results_dir, "plots")
        }
        
        # Create directories
        for dir_path in [self.results_dir] + list(self.dirs.values()):
            os.makedirs(dir_path, exist_ok=True)
            
        # Timer for tracking computation time
        self.metrics_timer = time.time()
        
        # Print confirmation
        print(f"Metrics will be saved to: {self.results_dir}")
        
        # Initialize metrics dictionaries
        self.reset_metrics()
        
        # Save configuration
        if config:
            with open(os.path.join(self.results_dir, "config.json"), 'w') as f:
                json.dump(self.convert_numpy_to_python(config), f, indent=2)
        
        # Track wall clock time and environment steps
        self.start_time = time.time()
        self.plot_interval = 10000  # Plot every 10k steps as in DCEO paper
    
    def reset_metrics(self):
        """Initialize/reset the metrics dictionaries."""
        
        # ===== Performance Metrics =====
        self.performance_metrics = {
            # Cumulative reward per episode
            'episode_rewards': [],
            
            # Learning curve: rewards by environment steps
            'env_steps': [],
            'reward_by_steps': [],
            'success_by_steps': [],
            
            # Sliding window performance within episodes
            'sliding_window_rewards': [],
            'window_step_indices': [],
            
            # Episode statistics
            'episode_lengths': [],
            'episode_success': [],
            'episodes_completed': 0,
            'total_env_steps': 0,
            
            # Per-10k-steps average return
            'step_milestones': [],
            'milestone_returns': []
        }
        
        # ===== Exploration Metrics =====
        self.exploration_metrics = {
            # State coverage over time (env_steps, coverage percentage)
            'step_coverage': [],
            
            # Track unique states visited (for computing coverage)
            'unique_states_visited': set(),
            
            # State visitation counts (for heatmap)
            'state_visitation': defaultdict(int),
            
            # Coverage after each episode
            'episode_coverage': [],
            
            # Goal reached metrics
            'goal_reached_steps': []
        }
        
        # Special handling for KeyDoorMazeEnv
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'num_keys'):
            self.exploration_metrics.update({
                'keys_collected': [False] * self.env.unwrapped.num_keys,
                'door_opened': [False] * len(self.env.unwrapped.door_positions),
                'keys_collected_steps': [],
                'doors_opened_steps': [],
            })
    
    def update_metrics(self, state, action, reward, next_state, done, info=None):
        """Update metrics based on a single environment step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information from the environment
        """
        info = info or {}
        self.performance_metrics['total_env_steps'] += 1
        
        # Track current episode reward in info
        if 'episode_reward' in info:
            current_episode_reward = info['episode_reward']
            # Add this to our tracked rewards for plotting even if episode isn't done
            # Only do this once every 10 steps to avoid duplicates
            if self.performance_metrics['total_env_steps'] % 10 == 0:
                # For incomplete episodes, track as ongoing rewards
                self.performance_metrics['reward_by_steps'].append(current_episode_reward)
                self.performance_metrics['env_steps'].append(self.performance_metrics['total_env_steps'])
        
        # Update state visitation for exploration metrics
        self._update_state_visitation(state)
        
        # Check for episode completion
        if done:
            self._update_episode_completion(reward, info)
        
        # Always update success tracking even for ongoing episodes
        # This helps when we have few episode completions but want to see progress
        if self.performance_metrics['total_env_steps'] % 50 == 0:
            # Force an update to the state coverage metrics
            if self.exploration_metrics['unique_states_visited']:
                total_maze_states = (self.maze_size - 2) ** 2  # Exclude walls
                coverage = len(self.exploration_metrics['unique_states_visited']) / total_maze_states
                self.exploration_metrics['step_coverage'].append((
                    self.performance_metrics['total_env_steps'],
                    coverage
                ))
                
                # Generate heatmap every 200 steps
                if self.performance_metrics['total_env_steps'] % 200 == 0:
                    self._generate_visitation_heatmap()
            
        # Periodic plotting and saving (reduce frequency for better performance)
        if self.performance_metrics['total_env_steps'] % self.plot_interval == 0:
            self.plot_all_metrics()
            self.save_all_metrics()
            
    def _update_state_visitation(self, state):
        """Update state visitation metrics.
        
        Args:
            state: Current state observation
        """
        # For maze environments, we care about the agent's position
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'agent_pos'):
            pos = tuple(self.env.unwrapped.agent_pos)
            
            # Update state visitation count
            self.exploration_metrics['state_visitation'][pos] += 1
            
            # Track unique states visited
            self.exploration_metrics['unique_states_visited'].add(pos)
            
            # Calculate and update coverage
            total_maze_states = (self.maze_size - 2) ** 2  # Exclude walls
            coverage = len(self.exploration_metrics['unique_states_visited']) / total_maze_states
            self.exploration_metrics['step_coverage'].append((
                self.performance_metrics['total_env_steps'],
                coverage
            ))
            
    def _update_episode_completion(self, reward, info):
        """Update metrics at episode completion.
        
        Args:
            reward: Final reward of episode
            info: Additional information from the environment
        """
        self.performance_metrics['episodes_completed'] += 1
        
        # Record episode success (positive reward at termination typically means success)
        success = reward > 0 or info.get('success', False)
        self.performance_metrics['episode_success'].append(success)
        
        # Update success-specific metrics
        if success:
            steps = info.get('episode_length', self.performance_metrics['total_env_steps'])
            self.exploration_metrics['goal_reached_steps'].append(steps)
            
        # Update key-door metrics if applicable
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'keys_collected'):
            keys_collected = self.env.unwrapped.keys_collected
            doors_opened = getattr(self.env.unwrapped, 'doors_opened', [])
            
            for i, key_collected in enumerate(keys_collected):
                if key_collected and not self.exploration_metrics['keys_collected'][i]:
                    self.exploration_metrics['keys_collected'][i] = True
                    self.exploration_metrics['keys_collected_steps'].append(
                        (i, self.performance_metrics['total_env_steps'])
                    )
                    
            for i, door_opened in enumerate(doors_opened):
                if door_opened and not self.exploration_metrics['door_opened'][i]:
                    self.exploration_metrics['door_opened'][i] = True
                    self.exploration_metrics['doors_opened_steps'].append(
                        (i, self.performance_metrics['total_env_steps'])
                    )
        
        # Update metrics at the end of episodes
        if hasattr(self.agent, 'episode_rewards') and len(self.agent.episode_rewards) > 0:
            episode_reward = self.agent.episode_rewards[-1]
            self.performance_metrics['episode_rewards'].append(episode_reward)
            
            # Update learning curve data
            self.performance_metrics['env_steps'].append(self.performance_metrics['total_env_steps'])
            self.performance_metrics['reward_by_steps'].append(episode_reward)
            self.performance_metrics['success_by_steps'].append(1 if success else 0)
            
            # Update episode length
            episode_length = info.get('episode_length', 0)
            self.performance_metrics['episode_lengths'].append(episode_length)
            
        # Calculate coverage after this episode
        if len(self.exploration_metrics['unique_states_visited']) > 0:
            total_maze_states = (self.maze_size - 2) ** 2  # Exclude walls
            coverage = len(self.exploration_metrics['unique_states_visited']) / total_maze_states
            self.exploration_metrics['episode_coverage'].append(coverage)
            
    def plot_all_metrics(self):
        """Generate and save all metric plots."""
        self._plot_performance_metrics()
        self._plot_exploration_metrics()
        
    def _plot_performance_metrics(self):
        """Generate and save performance metric plots."""
        plots_dir = self.dirs['plots']
        
        # Plot episode rewards
        if len(self.performance_metrics['episode_rewards']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.performance_metrics['episode_rewards'])
            plt.title(f'{self.agent_name} - Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'episode_rewards.png'))
            plt.close()
            
            # Plot moving average of rewards
            window_size = min(20, max(5, len(self.performance_metrics['episode_rewards']) // 10))
            if window_size > 0 and len(self.performance_metrics['episode_rewards']) > window_size:
                plt.figure(figsize=(10, 6))
                moving_avg = np.convolve(
                    self.performance_metrics['episode_rewards'], 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                plt.plot(moving_avg)
                plt.title(f'{self.agent_name} - Moving Average Reward (Window={window_size})')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, 'reward_moving_avg.png'))
                plt.close()
        
        # Plot learning curve (rewards by steps)
        if len(self.performance_metrics['env_steps']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.performance_metrics['env_steps'],
                self.performance_metrics['reward_by_steps'],
                marker='o', markersize=3, alpha=0.7
            )
            plt.title(f'{self.agent_name} - Reward vs Environment Steps')
            plt.xlabel('Environment Steps')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'reward_by_steps.png'))
            plt.close()
            
        # Plot success rate
        if len(self.performance_metrics['episode_success']) > 0:
            # Calculate success rate per 10 episodes
            success_rates = []
            episode_groups = list(range(0, len(self.performance_metrics['episode_success']), 10))
            
            for i in episode_groups:
                end_idx = min(i + 10, len(self.performance_metrics['episode_success']))
                group = self.performance_metrics['episode_success'][i:end_idx]
                success_rates.append(sum(group) / len(group) * 100)
                
            if success_rates:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    [i + 10 for i in episode_groups], 
                    success_rates, 
                    marker='o', markersize=5
                )
                plt.title(f'{self.agent_name} - Success Rate (per 10 episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Success Rate (%)')
                plt.ylim(0, 100)
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, 'success_rate.png'))
                plt.close()
                
        # Plot episode lengths
        if len(self.performance_metrics['episode_lengths']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.performance_metrics['episode_lengths'])
            plt.title(f'{self.agent_name} - Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'episode_lengths.png'))
            plt.close()
            
    def _plot_exploration_metrics(self):
        """Generate and save exploration metric plots."""
        plots_dir = self.dirs['plots']
        
        # Plot state coverage over time
        if self.exploration_metrics['step_coverage']:
            steps, coverage = zip(*self.exploration_metrics['step_coverage'])
            plt.figure(figsize=(10, 6))
            plt.plot(steps, [c * 100 for c in coverage])
            plt.title(f'{self.agent_name} - State Coverage Over Time')
            plt.xlabel('Environment Steps')
            plt.ylabel('Coverage (%)')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'state_coverage.png'))
            plt.close()
            
        # Generate state visitation heatmap
        if self.exploration_metrics['state_visitation']:
            self._generate_visitation_heatmap()
            
        # Plot goal reached timeline
        if self.exploration_metrics['goal_reached_steps']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.exploration_metrics['goal_reached_steps'], bins=20, alpha=0.7)
            plt.title(f'{self.agent_name} - Steps to Reach Goal')
            plt.xlabel('Environment Steps')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'goal_reached_histogram.png'))
            plt.close()
            
        # Plot key-door metrics for KeyDoorMazeEnv
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'num_keys'):
            self._plot_key_door_metrics()
            
    def _generate_visitation_heatmap(self):
        """Generate heatmap of state visitation counts."""
        plots_dir = self.dirs['plots']
        
        # Create a matrix to represent the maze
        heatmap = np.zeros((self.maze_size, self.maze_size))
        
        # Fill in visitation counts
        for (x, y), count in self.exploration_metrics['state_visitation'].items():
            if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
                heatmap[y, x] = count  # Note: y,x for proper orientation
                
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap, cmap='viridis', annot=False, cbar=True)
        plt.title(f'{self.agent_name} - State Visitation Heatmap')
        plt.savefig(os.path.join(plots_dir, 'visitation_heatmap.png'))
        plt.close()
        
    def _plot_key_door_metrics(self):
        """Plot key-door metrics for KeyDoorMazeEnv."""
        plots_dir = self.dirs['plots']
        
        # Plot key collection timeline
        if self.exploration_metrics['keys_collected_steps']:
            key_ids, key_steps = zip(*self.exploration_metrics['keys_collected_steps'])
            plt.figure(figsize=(10, 6))
            plt.bar(key_ids, key_steps)
            plt.title(f'{self.agent_name} - Steps to Collect Keys')
            plt.xlabel('Key ID')
            plt.ylabel('Environment Steps')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'key_collection_steps.png'))
            plt.close()
            
        # Plot door opened timeline
        if self.exploration_metrics['doors_opened_steps']:
            door_ids, door_steps = zip(*self.exploration_metrics['doors_opened_steps'])
            plt.figure(figsize=(10, 6))
            plt.bar(door_ids, door_steps)
            plt.title(f'{self.agent_name} - Steps to Open Doors')
            plt.xlabel('Door ID')
            plt.ylabel('Environment Steps')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'door_opening_steps.png'))
            plt.close()
    
    def save_all_metrics(self):
        """Save all metrics to JSON files."""
        # Save performance metrics
        self._save_metrics(
            self.performance_metrics,
            os.path.join(self.dirs['performance'], 'performance_metrics.json')
        )
        
        # Save exploration metrics (excluding sets)
        exploration_metrics = {
            k: v for k, v in self.exploration_metrics.items()
            if not isinstance(v, set)
        }
        # Convert defaultdict to regular dict for JSON serialization
        exploration_metrics['state_visitation'] = dict(exploration_metrics['state_visitation'])
        
        self._save_metrics(
            exploration_metrics,
            os.path.join(self.dirs['exploration'], 'exploration_metrics.json')
        )
            
    def _save_metrics(self, metrics_dict, filename):
        """Save metrics dictionary to JSON file.
        
        Args:
            metrics_dict: Dictionary of metrics to save
            filename: Output JSON filename
        """
        # Convert numpy/torch types to Python native types
        serializable_dict = self.convert_numpy_to_python(metrics_dict)
        
        with open(filename, 'w') as f:
            json.dump(serializable_dict, f, indent=2)
            
    def convert_numpy_to_python(self, obj):
        """Convert numpy/torch values to Python native types for JSON serialization.
        
        Args:
            obj: Object to convert (can be a single value, list, or dictionary)
            
        Returns:
            Converted object with numpy/torch values replaced by Python native types
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_python(val) for key, val in obj.items()}
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_to_python(val) for val in obj)
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(val) for val in obj]
        elif isinstance(obj, (str, bool, int, float, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            return self.convert_numpy_to_python(obj.__dict__)
        else:
            try:
                # Try to convert to a simpler type
                simple_obj = float(obj) if hasattr(obj, '__float__') else str(obj)
                return simple_obj
            except:
                return str(obj)
