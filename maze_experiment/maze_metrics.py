"""
Complex Maze Metrics - DCEO Paper Implementation


This module implements the following metrics categories from the paper:
- Performance Metrics (section 4.5.1)
- Exploration Metrics (section 4.5.2) 
- Option Quality Metrics (section 4.5.3)
- Representation Quality Metrics (section 4.5.4)

Metrics are tracked continuously during training and visualized as in the original paper.
"""

import os
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# Import representation quality metrics extension
from maze_metrics_extensions import RepresentationQualityMetrics

class MazeMetricsTracker:
    """
    Metrics tracker for Complex Maze environment following DCEO paper methodology.
    Implements the metrics specified in sections 4.5.1, 4.5.2, and 4.5.3 of the paper.
    """
    
    def __init__(self, env, agent, config=None):
        """Initialize the metrics tracker.
        
        Args:
            env: The Complex Maze environment
            agent: The DCEO agent
            config: Configuration dictionary with training parameters
        """
        self.env = env
        self.agent = agent
        self.config = config or {}
        
        # Create timestamp for this training run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory for this run
        self.results_dir = os.path.join("maze_results", f"run_{self.timestamp}")
        
        # Create subdirectories for different metric categories
        self.dirs = {
            'performance': os.path.join(self.results_dir, "performance_metrics"),
            'exploration': os.path.join(self.results_dir, "exploration_metrics"),
            'options': os.path.join(self.results_dir, "option_metrics"),
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
        
        # Initialize representation quality metrics extension
        self.rep_extension = RepresentationQualityMetrics(self)
        
        # Save configuration
        if config:
            with open(os.path.join(self.results_dir, "config.json"), 'w') as f:
                json.dump(self.convert_numpy_to_python(config), f, indent=2)
        
        # Track wall clock time and environment steps
        self.start_time = time.time()
        self.plot_interval = 10000  # Plot every 10k steps as in DCEO paper
    
    def reset_metrics(self):
        """Initialize/reset the metrics dictionaries following DCEO paper methodology."""
        
        # ===== 4.5.1 Performance Metrics =====
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
            'episode_steps': [],
            'episode_success': [],
            'episodes_completed': 0,
            'total_steps': 0,
            
            # Per-10k-steps average return
            'step_milestones': [],
            'milestone_returns': [],
            
            # Wall clock time tracking
            'wall_time': []
        }
        
        # ===== 4.5.2 Exploration Metrics =====
        self.exploration_metrics = {
            # State coverage over time (env_steps, coverage percentage)
            'step_coverage': [],
            
            # Track unique states visited (for computing coverage)
            'unique_states_visited': set(),
            'visited_positions': set(),
            
            # State visitation counts (for heatmap)
            'state_visitation': defaultdict(int),
            
            # Coverage after each episode
            'episode_coverage': [],
            'coverage': []
        }
        
        # ===== 4.5.3 Option Quality Metrics =====
        self.option_metrics = {
            # Option usage frequency
            'option_frequency': defaultdict(int),
            'primitive_action_count': 0,
            
            # Option behaviors
            'option_trajectories': {i: [] for i in range(self.agent.num_options)},
            
            # State visitation by each option (for overlap analysis)
            'option_state_visitation': {i: defaultdict(int) for i in range(self.agent.num_options)},
            
            # Duration tracking for each option
            'option_durations': defaultdict(list),
            
            # Displacement caused by each option
            'option_displacements': defaultdict(list),
            
            # Current active option (for tracking)
            'current_option': None,
            'option_start_state': None,
            'option_start_step': 0,
            'option_steps': defaultdict(int)
        }
        
        # ===== Continuous Monitoring =====
        self.continuous_metrics = {
            'steps': [],           # Environment steps for plotting
            'rewards': [],         # Smoothed rewards
            'success_rates': [],   # Success rates
            'episode_lengths': [], # Average episode lengths
            'state_coverage': [],  # State coverage percentage
            'unique_states': [],   # Number of unique states visited
            'wall_time': []        # Wall clock time for performance monitoring
        }
        
        # Current episode tracking
        self.current_episode = {
            'steps': 0,
            'reward': 0,
            'states_visited': set(),
            'success': False
        }
    
    def _log_performance_metrics(self, reward, done, step_count):
        """Log basic performance metrics."""
        # This function intentionally left empty as logging happens in log_episode_end
        # We keep this function to maintain the consistent interface
        pass
    
    def _log_representation_metrics(self, state, next_state, step_count):
        """Log representation quality metrics using the extension."""
        if hasattr(self, 'rep_extension'):
            self.rep_extension.log_representation_metrics(state, next_state, step_count)
    
    def log_training_step(self, state, action, reward, next_state, done, info, step_count, episode_step):
        """Log metrics for a single training step following DCEO paper methodology.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information
            step_count: Global step counter (env interactions)
            episode_step: Steps in current episode
        """
        # Update total environment steps
        self.performance_metrics['total_steps'] = step_count
        
        # Track current episode stats
        self.current_episode['steps'] += 1
        self.current_episode['reward'] += reward
        
        # Check for success/goal reached
        if info.get('goal_reached', False):
            self.current_episode['success'] = True
        
        # Track agent position
        agent_pos = None
        if hasattr(self.env.unwrapped, 'agent_position'):
            agent_pos = tuple(self.env.unwrapped.agent_position)
        else:
            agent_pos = self._extract_agent_position(state)
        
        if agent_pos:
            # Add to visited positions
            self.exploration_metrics['visited_positions'].add(agent_pos)
            self.exploration_metrics['unique_states_visited'].add(agent_pos)
            self.current_episode['states_visited'].add(agent_pos)
            
            # Update state visitation heatmap
            self.exploration_metrics['state_visitation'][agent_pos] += 1
            
            # Option-specific logging
            if hasattr(self.agent, 'cur_opt') and self.agent.cur_opt is not None:
                opt_id = self.agent.cur_opt
                
                # Track option state visitation
                self.option_metrics['option_state_visitation'][opt_id][agent_pos] += 1
                
                # Track option transitions
                if self.option_metrics['current_option'] is None:
                    # Starting new option
                    self.option_metrics['current_option'] = opt_id
                    self.option_metrics['option_start_state'] = agent_pos
                    self.option_metrics['option_start_step'] = step_count
                    self.option_metrics['option_frequency'][opt_id] += 1
                elif self.option_metrics['current_option'] != opt_id:
                    # Option switch detected
                    prev_opt = self.option_metrics['current_option']
                    prev_start = self.option_metrics['option_start_state']
                    prev_step = self.option_metrics['option_start_step']
                    
                    # Record duration
                    duration = step_count - prev_step
                    self.option_metrics['option_durations'][prev_opt].append(duration)
                    self.option_metrics['option_steps'][prev_opt] += duration
                    
                    # Record displacement
                    if prev_start and agent_pos:
                        dx = agent_pos[0] - prev_start[0]
                        dy = agent_pos[1] - prev_start[1]
                        displacement = np.sqrt(dx**2 + dy**2)
                        self.option_metrics['option_displacements'][prev_opt].append(displacement)
                    
                    # Start new option
                    self.option_metrics['current_option'] = opt_id
                    self.option_metrics['option_start_state'] = agent_pos
                    self.option_metrics['option_start_step'] = step_count
                    self.option_metrics['option_frequency'][opt_id] += 1
            else:
                # Primitive action
                self.option_metrics['primitive_action_count'] += 1
        
        # Log representation metrics
        self._log_representation_metrics(state, next_state, step_count)
        
        # Check for plot generation
        if step_count % self.plot_interval == 0:
            self._generate_milestone_plots(step_count)
            self.metrics_timer = time.time()
        
        # When episode terminates, call log_episode_end
        if done:
            self.log_episode_end(
                self.current_episode['reward'],
                self.current_episode['steps'],
                step_count // self.plot_interval
            )
            
            # Reset current episode tracking
            self.current_episode = {
                'steps': 0,
                'reward': 0,
                'states_visited': set(),
                'success': False
            }
    
    def log_episode_end(self, episode_reward, episode_steps, iteration):
        """Log metrics at the end of an episode following DCEO paper methodology.
        
        Args:
            episode_reward: Total reward for the episode
            episode_steps: Number of steps in the episode
            iteration: Current iteration number
        """
        # Record episode statistics
        self.performance_metrics['episodes_completed'] += 1
        
        # Record episode statistics
        episode_reward = self.current_episode['reward']
        success = self.current_episode['success']
        steps = self.current_episode['steps']
        self.get_episode_metrics(episode_reward, success, steps)
        
        # Record coverage for this episode
        coverage = len(self.current_episode['states_visited'])
        if hasattr(self.env.unwrapped, 'maze_size'):
            maze_size = self.env.unwrapped.maze_size
            coverage_pct = coverage / (maze_size * maze_size) * 100.0
            self.exploration_metrics['episode_coverage'].append(coverage_pct)
    
    def _generate_milestone_plots(self, step_count):
        """Generate plots at milestone steps (e.g., every 10k steps) as in the DCEO paper.
        
        Args:
            step_count: Current environment step count
        """
        # Record wall clock time
        elapsed = time.time() - self.start_time
        self.performance_metrics['wall_time'].append((step_count, elapsed))
        
        # Skip if no episodes have been completed
        if self.performance_metrics['episodes_completed'] == 0:
            return
        
        # Generate plots based on DCEO paper methodology
        self.plot_metrics(step_count)
    
    def get_episode_metrics(self, episode_reward, success, steps):
        """Log episode completion metrics and return the metrics."""
        # Record rewards
        self.performance_metrics['episode_rewards'].append(episode_reward)
        self.performance_metrics['episode_success'].append(1.0 if success else 0.0)
        self.performance_metrics['episode_steps'].append(steps)
        
        # Calculate coverage
        coverage = 0.0
        num_unique_states = 0
        if hasattr(self.env.unwrapped, 'get_coverage_stats'):
            stats = self.env.unwrapped.get_coverage_stats()
            coverage = stats.get('coverage', 0.0)
            num_unique_states = stats.get('unique_states', len(self.exploration_metrics['visited_positions']))
            self.exploration_metrics['coverage'].append(coverage)
        
        # Update episode count
        self.performance_metrics['episodes_completed'] += 1
        
        # Update total steps
        self.performance_metrics['total_steps'] += steps
        total_steps = self.performance_metrics['total_steps']
        
        # Update continuous metrics for plotting
        self.continuous_metrics['steps'].append(total_steps)
        
        # Calculate moving averages for continuous metrics
        window = min(10, self.performance_metrics['episodes_completed'])
        recent_rewards = self.performance_metrics['episode_rewards'][-window:]
        recent_success = self.performance_metrics['episode_success'][-window:]
        recent_steps = self.performance_metrics['episode_steps'][-window:]
        
        self.continuous_metrics['rewards'].append(np.mean(recent_rewards))
        self.continuous_metrics['success_rates'].append(np.mean(recent_success))
        self.continuous_metrics['episode_lengths'].append(np.mean(recent_steps))
        self.continuous_metrics['state_coverage'].append(coverage)
        self.continuous_metrics['unique_states'].append(num_unique_states)
        
        # Return current metrics
        return {
            'reward': episode_reward,
            'success': success,
            'steps': steps,
            'coverage': coverage,
            'unique_states': num_unique_states
        }
    
    def get_training_summary(self, num_episodes=10):
        """Get summary of recent training episodes."""
        if len(self.performance_metrics['episode_rewards']) == 0:
            return {
                'avg_reward': 0.0,
                'success_rate': 0.0,
                'avg_steps': 0.0,
                'coverage': 0.0
            }
        
        recent_rewards = self.performance_metrics['episode_rewards'][-num_episodes:]
        recent_success = self.performance_metrics['episode_success'][-num_episodes:]
        recent_steps = self.performance_metrics['episode_steps'][-num_episodes:]
        
        # Calculate coverage
        coverage = 0.0
        if self.exploration_metrics['coverage']:
            # Get latest coverage percentage
            coverage = self.exploration_metrics['coverage'][-1]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'success_rate': np.mean(recent_success),
            'avg_steps': np.mean(recent_steps),
            'coverage': coverage
        }
    
    def save_metrics(self):
        """Save all metrics to files following DCEO paper structure."""
        # Save performance metrics
        with open(os.path.join(self.dirs['performance'], 'performance_metrics.json'), 'w') as f:
            json.dump(self.convert_numpy_to_python(self.performance_metrics), f, indent=2)
        
        # Save exploration metrics
        with open(os.path.join(self.dirs['exploration'], 'exploration_metrics.json'), 'w') as f:
            json.dump(self.convert_numpy_to_python(self.exploration_metrics), f, indent=2)
        
        # Save option metrics
        with open(os.path.join(self.dirs['options'], 'option_metrics.json'), 'w') as f:
            json.dump(self.convert_numpy_to_python(self.option_metrics), f, indent=2)
        
        # Save continuous metrics
        with open(os.path.join(self.results_dir, 'continuous_metrics.json'), 'w') as f:
            json.dump(self.convert_numpy_to_python(self.continuous_metrics), f, indent=2)
    
    def plot_metrics(self, step_count, final=False):
        """Plot all metrics at current step count."""
        # Performance metrics plots
        self._plot_performance_metrics(step_count)
        
        # Exploration metrics plots
        self._plot_exploration_metrics(step_count)
        
        # Option metrics plots
        self._plot_option_metrics(step_count)
        
        # If final plot, create summary plots
        if final:
            self._create_summary_plots()
    
    def _create_summary_plots(self):
        """Create summary plots of all metrics at the end of training."""
        # Get data for plotting
        steps = self.continuous_metrics['steps']
        if not steps:
            return
        
        # Performance summary
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(steps, self.continuous_metrics['rewards'], 'b-')
        plt.xlabel('Environment Steps')
        plt.ylabel('Average Reward')
        plt.title('Training Rewards')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(steps, self.continuous_metrics['success_rates'], 'g-')
        plt.xlabel('Environment Steps')
        plt.ylabel('Success Rate')
        plt.title('Success Rate')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(steps, self.continuous_metrics['episode_lengths'], 'r-')
        plt.xlabel('Environment Steps')
        plt.ylabel('Average Episode Length')
        plt.title('Episode Lengths')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(steps, self.continuous_metrics['state_coverage'], 'c-')
        plt.xlabel('Environment Steps')
        plt.ylabel('State Coverage (%)')
        plt.title('State Coverage')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'summary_performance.png'))
        plt.close()
        
        # Option usage summary
        if hasattr(self.agent, 'num_options') and self.agent.num_options > 0:
            plt.figure(figsize=(12, 8))
            option_frequency = self.option_metrics['option_frequency']
            
            # Convert option frequency to percentages
            options = list(option_frequency.keys())
            frequencies = [option_frequency[opt] for opt in options]
            total = sum(frequencies)
            if total > 0:
                percentages = [100 * freq / total for freq in frequencies]
                
                plt.bar(options, percentages)
                plt.xlabel('Option ID')
                plt.ylabel('Usage Percentage (%)')
                plt.title('Option Usage Distribution')
                plt.grid(True)
                plt.savefig(os.path.join(self.dirs['options'], 'option_usage_summary.png'))
                plt.close()
    
    def _plot_exploration_metrics(self, step_count):
        """Plot exploration metrics."""
        # Get data for plotting
        steps = self.continuous_metrics['steps']
        if not steps:
            return
        
        # State coverage plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.continuous_metrics['state_coverage'], 'c-')
        plt.xlabel('Environment Steps')
        plt.ylabel('State Coverage (%)')
        plt.title(f'State Coverage at Step {step_count}')
        plt.grid(True)
        plt.savefig(os.path.join(self.dirs['exploration'], f'state_coverage_{step_count}.png'))
        plt.close()
        
        # Unique states visited plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.continuous_metrics['unique_states'], 'm-')
        plt.xlabel('Environment Steps')
        plt.ylabel('Unique States Visited')
        plt.title(f'Unique States Visited at Step {step_count}')
        plt.grid(True)
        plt.savefig(os.path.join(self.dirs['exploration'], f'unique_states_{step_count}.png'))
        plt.close()
        
        # Generate heatmap of visitation frequency
        self._plot_visitation_heatmap(step_count)
    
    def _plot_visitation_heatmap(self, step_count):
        """Generate heatmap of state visitation frequency."""
        if not self.exploration_metrics['state_visitation']:
            return
            
        if not hasattr(self.env.unwrapped, 'maze_size'):
            return
            
        # Get maze dimensions
        maze_size = self.env.unwrapped.maze_size
        
        # Create grid for heatmap
        visitation_grid = np.zeros((maze_size, maze_size))
        
        # Fill grid with visitation counts
        for pos, count in self.exploration_metrics['state_visitation'].items():
            if len(pos) == 2:
                x, y = pos
                if 0 <= x < maze_size and 0 <= y < maze_size:
                    visitation_grid[y, x] = count
        
        # Apply log scale for better visualization (1 is added to avoid log(0))
        with np.errstate(divide='ignore'):
            log_grid = np.log1p(visitation_grid)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(log_grid, cmap='viridis', cbar=True)
        plt.title(f'State Visitation Heatmap at Step {step_count}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['exploration'], f'heatmap_{step_count}.png'))
        plt.close()
    
    def _plot_option_metrics(self, step_count):
        """Plot option usage metrics."""
        # Skip if agent doesn't have options
        if not hasattr(self.agent, 'num_options') or self.agent.num_options <= 0:
            return
            
        # Option frequency plot
        option_frequency = self.option_metrics['option_frequency']
        if option_frequency:
            plt.figure(figsize=(10, 6))
            
            # Convert option frequency to percentages
            options = list(option_frequency.keys())
            frequencies = [option_frequency[opt] for opt in options]
            total = sum(frequencies)
            if total > 0:
                percentages = [100 * freq / total for freq in frequencies]
                
                plt.bar(options, percentages)
                plt.xlabel('Option ID')
                plt.ylabel('Usage Percentage (%)')
                plt.title(f'Option Usage Distribution at Step {step_count}')
                plt.grid(True)
                plt.savefig(os.path.join(self.dirs['options'], f'option_usage_{step_count}.png'))
                plt.close()
        
        # Option duration plot
        if self.option_metrics['option_durations']:
            plt.figure(figsize=(10, 6))
            
            # Average duration per option
            options = list(self.option_metrics['option_durations'].keys())
            durations = [np.mean(self.option_metrics['option_durations'][opt]) if self.option_metrics['option_durations'][opt] else 0 
                        for opt in options]
            
            plt.bar(options, durations)
            plt.xlabel('Option ID')
            plt.ylabel('Average Duration (steps)')
            plt.title(f'Option Duration Distribution at Step {step_count}')
            plt.grid(True)
            plt.savefig(os.path.join(self.dirs['options'], f'option_duration_{step_count}.png'))
            plt.close()
    
    def _plot_performance_metrics(self, step_count):
        """Plot performance metrics."""
        # Get data for plotting
        steps = self.continuous_metrics['steps']
        if not steps:
            return
        
        # Rewards plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.continuous_metrics['rewards'], 'b-')
        plt.xlabel('Environment Steps')
        plt.ylabel('Average Reward')
        plt.title(f'Training Rewards at Step {step_count}')
        plt.grid(True)
        plt.savefig(os.path.join(self.dirs['performance'], f'rewards_{step_count}.png'))
        plt.close()
        
        # Success rate plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.continuous_metrics['success_rates'], 'g-')
        plt.xlabel('Environment Steps')
        plt.ylabel('Success Rate')
        plt.title(f'Success Rate at Step {step_count}')
        plt.grid(True)
        plt.savefig(os.path.join(self.dirs['performance'], f'success_rate_{step_count}.png'))
        plt.close()
        
        # Episode length plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.continuous_metrics['episode_lengths'], 'r-')
        plt.xlabel('Environment Steps')
        plt.ylabel('Average Episode Length')
        plt.title(f'Episode Lengths at Step {step_count}')
        plt.grid(True)
        plt.savefig(os.path.join(self.dirs['performance'], f'episode_length_{step_count}.png'))
        plt.close()
    
    def _extract_agent_position(self, state):
        """Extract agent position from state representation.
        
        This is a fallback method if the agent's position isn't directly accessible.
        
        Args:
            state: State observation
            
        Returns:
            Tuple (x, y) of agent position, or None if not extractable
        """
        # Try to extract position from state
        # This is environment-specific and may need adaptation
        try:
            if hasattr(self.env.unwrapped, 'agent_position'):
                return tuple(self.env.unwrapped.agent_position)
            
            # If state is a 2D grid (common in maze environments)
            if isinstance(state, np.ndarray) and state.ndim == 2:
                # Find agent marker (usually 2 or similar value)
                agent_pos = np.where(state == 2)
                if len(agent_pos[0]) > 0 and len(agent_pos[1]) > 0:
                    return (agent_pos[1][0], agent_pos[0][0])  # (x, y)
            
            # If state is a flattened vector with one-hot encoding
            elif isinstance(state, np.ndarray) and state.ndim == 1:
                # This depends on how the maze state is encoded
                # Need to be adapted based on the specific environment
                pass
        except:
            pass
            
        return None
    
    def convert_numpy_to_python(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Convert tuple keys to strings for JSON serialization
                if isinstance(k, tuple):
                    k = str(k)
                result[k] = self.convert_numpy_to_python(v)
            return result
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_to_python(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self.convert_numpy_to_python(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, set):
            return list(self.convert_numpy_to_python(item) for item in obj)
        else:
            return obj
