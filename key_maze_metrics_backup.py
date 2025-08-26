"""
Key Maze Metrics - DCEO Paper Implementation
Implementing metrics as described in "Deep Covering Options: An Eigenfunction View of Successor Features"
by Klissarov et al. (2023).

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

class KeyMazeMetricsTracker:
    """
    Metrics tracker for Key Maze environment following DCEO paper methodology.
    Implements the metrics specified in sections 4.5.1, 4.5.2, and 4.5.3 of the paper.
    """
    
    def __init__(self, env, agent, config=None):
        """Initialize the metrics tracker.
        
        Args:
            env: The Key Maze environment
            agent: The DCEO agent
            config: Configuration dictionary with training parameters
        """
        self.env = env
        self.agent = agent
        self.config = config or {}
        
        # Create timestamp for this training run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory for this run
        self.results_dir = os.path.join("key_maze_results", f"run_{self.timestamp}")
        
        # Create subdirectories for different metric categories
        self.dirs = {
            'performance': os.path.join(self.results_dir, "performance_metrics"),
            'exploration': os.path.join(self.results_dir, "exploration_metrics"),
            'option': os.path.join(self.results_dir, "option_metrics"),
            'plots': os.path.join(self.results_dir, "plots")
        }
        
        # Create directories
        for dir_path in [self.results_dir] + list(self.dirs.values()):
            os.makedirs(dir_path, exist_ok=True)
        
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
            'episode_lengths': [],
            'episode_success': [],
            'episodes_completed': 0,
            'total_env_steps': 0,
            
            # Per-10k-steps average return
            'step_milestones': [],
            'milestone_returns': []
        }
        
        # ===== 4.5.2 Exploration Metrics =====
        self.exploration_metrics = {
            # State coverage over time (env_steps, coverage percentage)
            'step_coverage': [],
            
            # Track unique states visited (for computing coverage)
            'unique_states_visited': set(),
            
            # State visitation counts (for heatmap)
            'state_visitation': defaultdict(int),
            
            # Coverage after each episode
            'episode_coverage': [],
            
            # Key-specific metrics
            'keys_collected': [False] * self.env.unwrapped.num_keys,
            'door_opened': [False] * len(self.env.unwrapped.door_positions),
            'keys_collected_steps': [],
            'doors_opened_steps': [],
            'goal_reached_steps': []
        }
        
        # ===== 4.5.3 Option Quality Metrics =====
        self.option_metrics = {
            # Option usage frequency
            'option_selection_counts': defaultdict(int),
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
        
        # ===== Continuous Monitoring (as in paper) =====
        self.continuous_metrics = {
            # Record metrics at regular intervals (every 10k steps)
            'step_milestones': [],
            'coverage_at_milestones': [],
            'option_usage_at_milestones': [],
            'reward_at_milestones': []
        }
        
        # Current episode tracking
        self.current_episode = {
            'steps': 0,
            'reward': 0,
            'states_visited': set(),
            'success': False
        }
    
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
        self.performance_metrics['total_env_steps'] = step_count
        
        # Update learning curve data for plotting
        self.performance_metrics['env_steps'].append(step_count)
        
        # Calculate current sliding window metrics
        if len(self.performance_metrics['episode_rewards']) > 0:
            window_size = min(5, len(self.performance_metrics['episode_rewards']))
            recent_rewards = self.performance_metrics['episode_rewards'][-window_size:]
            avg_reward = sum(recent_rewards) / window_size
            self.performance_metrics['reward_by_steps'].append(avg_reward)
            
            # Also track success rate
            if len(self.performance_metrics['episode_success']) >= window_size:
                recent_success = self.performance_metrics['episode_success'][-window_size:]
                success_rate = sum(recent_success) / window_size
                self.performance_metrics['success_by_steps'].append(success_rate)
            else:
                self.performance_metrics['success_by_steps'].append(0.0)
        else:
            # No episodes completed yet
            self.performance_metrics['reward_by_steps'].append(0.0)
            self.performance_metrics['success_by_steps'].append(0.0)
        
        # Update current episode tracking
        self.current_episode['steps'] = episode_step
        self.current_episode['reward'] += reward
        if info.get('goal_reached', False):
            self.current_episode['success'] = True
        
        # ===== 4.5.2 Exploration Metrics - State Coverage/Visitation =====
        if hasattr(self.env.unwrapped, 'agent_position'):
            state_pos = tuple(self.env.unwrapped.agent_position)
        else:
            state_pos = self._extract_agent_position(state)
        
        if state_pos:
            # Add to unique states visited (both global and for current episode)
            self.exploration_metrics['unique_states_visited'].add(state_pos)
            self.current_episode['states_visited'].add(state_pos)
            
            # Update state visitation count (for heatmap)
            self.exploration_metrics['state_visitation'][state_pos] += 1
            
            # Record state coverage at regular intervals (every 10k steps)
            if step_count % self.plot_interval == 0 or step_count == 1:
                total_states = self.env.unwrapped.maze_size * self.env.unwrapped.maze_size
                coverage = len(self.exploration_metrics['unique_states_visited']) / total_states
                self.exploration_metrics['step_coverage'].append((step_count, coverage))
                
                # Also update continuous metrics
                self.continuous_metrics['step_milestones'].append(step_count)
                self.continuous_metrics['coverage_at_milestones'].append(coverage)
                
                # Time to generate plots based on the paper's continuous monitoring approach
                self._generate_milestone_plots(step_count)
        
        # ===== 4.5.3 Option Quality Metrics =====
        current_option = self.agent.cur_opt
        
        # Track option usage frequency
        if current_option is not None:
            # If this is a new option or transition from primitive to option
            if self.option_metrics['current_option'] != current_option:
                # Count this as a new option selection
                self.option_metrics['option_selection_counts'][current_option] += 1
                
                # Store option start state for displacement calculation
                self.option_metrics['option_start_state'] = state_pos
                self.option_metrics['option_start_step'] = step_count
                
                # If previous option was active, record its duration
                if self.option_metrics['current_option'] is not None:
                    prev_option = self.option_metrics['current_option']
                    duration = step_count - self.option_metrics['option_start_step']
                    self.option_metrics['option_durations'][prev_option].append(duration)
                    
                    # Calculate displacement if we have both start and end positions
                    if self.option_metrics['option_start_state'] and state_pos:
                        start_x, start_y = self.option_metrics['option_start_state']
                        end_x, end_y = state_pos
                        displacement = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
                        self.option_metrics['option_displacements'][prev_option].append(displacement)
                
                # Update current option
                self.option_metrics['current_option'] = current_option
            
            # Add current state to this option's trajectory
            if state_pos:
                self.option_metrics['option_trajectories'][current_option].append(state_pos)
                self.option_metrics['option_state_visitation'][current_option][state_pos] += 1
            
            # Increment steps counter for this option
            self.option_metrics['option_steps'][current_option] += 1
        else:
            # Using primitive action
            self.option_metrics['primitive_action_count'] += 1
            
            # If transitioning from option to primitive, record option metrics
            if self.option_metrics['current_option'] is not None:
                prev_option = self.option_metrics['current_option']
                duration = step_count - self.option_metrics['option_start_step']
                self.option_metrics['option_durations'][prev_option].append(duration)
                
                # Calculate displacement
                if self.option_metrics['option_start_state'] and state_pos:
                    start_x, start_y = self.option_metrics['option_start_state']
                    end_x, end_y = state_pos
                    displacement = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
                    self.option_metrics['option_displacements'][prev_option].append(displacement)
                
                # Reset current option
                self.option_metrics['current_option'] = None
        
        # ===== Record option usage at milestones =====
        if step_count % self.plot_interval == 0 or step_count == 1:
            option_usage = {i: self.option_metrics['option_selection_counts'][i] 
                          for i in range(self.agent.num_options)}
            primitive_count = self.option_metrics['primitive_action_count']
            option_usage['primitive'] = primitive_count
            self.continuous_metrics['option_usage_at_milestones'].append(option_usage)
        
        # ===== 4.5.1 Performance Metrics - Learning Curve =====
        # Record reward at regular intervals for learning curve
        if step_count % self.plot_interval == 0 or step_count == 1:
            # Calculate average reward over last window
            if len(self.performance_metrics['episode_rewards']) > 0:
                # Average the last 5 episodes or all available episodes
                window_size = min(5, len(self.performance_metrics['episode_rewards']))
                avg_reward = sum(self.performance_metrics['episode_rewards'][-window_size:]) / window_size
                self.performance_metrics['reward_by_steps'].append(avg_reward)
                self.performance_metrics['env_steps'].append(step_count)
                
                # Also calculate success rate
                if len(self.performance_metrics['episode_success']) >= window_size:
                    success_rate = sum(self.performance_metrics['episode_success'][-window_size:]) / window_size
                    self.performance_metrics['success_by_steps'].append(success_rate)
                
                # Update continuous monitoring
                self.continuous_metrics['reward_at_milestones'].append(avg_reward)
        
        # ===== Environment-Specific Events =====
        # Track key collection
        if info.get('key_collected', False):
            key_id = info.get('key_id', 0)
            if not self.exploration_metrics['keys_collected'][key_id]:
                self.exploration_metrics['keys_collected'][key_id] = True
                self.exploration_metrics['keys_collected_steps'].append((step_count, episode_step))
        
        # Track door opening
        if info.get('door_opened', False):
            door_id = info.get('door_id', 0)
            if not self.exploration_metrics['door_opened'][door_id]:
                self.exploration_metrics['door_opened'][door_id] = True
                self.exploration_metrics['doors_opened_steps'].append((step_count, episode_step))
        
        # Track goal reaching
        if info.get('goal_reached', False):
            self.exploration_metrics['goal_reached_steps'].append((step_count, episode_step))
    
    def log_episode_end(self, episode_reward, episode_steps, iteration):
        """Log metrics at the end of an episode following DCEO paper methodology.
        
        Args:
            episode_reward: Total reward for the episode
            episode_steps: Number of steps in the episode
            iteration: Current iteration number
        """
        # Update episode counter
        self.performance_metrics['episodes_completed'] += 1
        
        # Record episode metrics
        self.performance_metrics['episode_rewards'].append(episode_reward)
        self.performance_metrics['episode_lengths'].append(episode_steps)
        
        # Record whether episode was successful
        self.performance_metrics['episode_success'].append(self.current_episode['success'])
        
        # Compute state coverage percentage for this episode
        total_states = self.env.unwrapped.maze_size * self.env.unwrapped.maze_size
        episode_coverage = len(self.current_episode['states_visited']) / total_states
        self.exploration_metrics['episode_coverage'].append((self.performance_metrics['episodes_completed'], episode_coverage))
        
        # Compute global coverage
        global_coverage = len(self.exploration_metrics['unique_states_visited']) / total_states
        
        # Track 10k-step returns as in the paper
        current_step = self.performance_metrics['total_env_steps']
        if current_step % self.plot_interval == 0 or current_step == 0:
            # Calculate average return over last several episodes
            window_size = min(5, len(self.performance_metrics['episode_rewards']))
            avg_return = sum(self.performance_metrics['episode_rewards'][-window_size:]) / window_size
            self.performance_metrics['step_milestones'].append(current_step)
            self.performance_metrics['milestone_returns'].append(avg_return)
            
            # Also store option usage stats at this milestone
            option_usage = {}
            for opt_idx in range(self.agent.num_options):
                option_usage[opt_idx] = self.option_metrics['option_selection_counts'].get(opt_idx, 0)
            option_usage['primitive'] = self.option_metrics['primitive_action_count']
            self.continuous_metrics['option_usage_at_milestones'].append(option_usage)
            self.continuous_metrics['reward_at_milestones'].append(avg_return)
        
        # Reset current episode tracking
        self.current_episode = {
            'steps': 0,
            'reward': 0,
            'states_visited': set(),
            'success': False
        }
    
    def _generate_milestone_plots(self, step_count):
        """Generate plots at milestone steps (e.g., every 10k steps) as in the DCEO paper.
        
        Args:
            step_count: Current environment step count
        """
        suffix = f"_step_{step_count}"
        
        # Create plots directory
        if not os.path.exists(self.dirs['plots']):
            os.makedirs(self.dirs['plots'], exist_ok=True)
        
        # 1. Learning Curve (reward vs environment steps)
        if len(self.performance_metrics['env_steps']) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.performance_metrics['env_steps'], self.performance_metrics['reward_by_steps'], 'b-', linewidth=2)
            plt.title('Learning Curve: Average Return vs Environment Steps')
            plt.xlabel('Environment Steps')
            plt.ylabel('Average Return')
            plt.grid(True)
            plt.savefig(os.path.join(self.dirs['plots'], f"learning_curve{suffix}.png"))
            plt.close()
            
            # Success rate curve
            if len(self.performance_metrics['success_by_steps']) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(self.performance_metrics['env_steps'], self.performance_metrics['success_by_steps'], 'g-', linewidth=2)
                plt.title('Success Rate vs Environment Steps')
                plt.xlabel('Environment Steps')
                plt.ylabel('Success Rate')
                plt.grid(True)
                plt.savefig(os.path.join(self.dirs['plots'], f"success_rate{suffix}.png"))
                plt.close()
        
        # 2. State Coverage over time
        if len(self.exploration_metrics['step_coverage']) > 1:
            plt.figure(figsize=(10, 6))
            steps, coverages = zip(*self.exploration_metrics['step_coverage'])
            plt.plot(steps, coverages, 'g-', linewidth=2)
            plt.title('State Space Coverage vs Environment Steps')
            plt.xlabel('Environment Steps')
            plt.ylabel('Coverage (%)')
            plt.grid(True)
            plt.savefig(os.path.join(self.dirs['plots'], f"state_coverage{suffix}.png"))
            plt.close()
            
        # 3. Option Usage Frequency
        if self.continuous_metrics['option_usage_at_milestones']:
            usage_data = self.continuous_metrics['option_usage_at_milestones'][-1]  # Get latest usage data
            options = sorted([k for k in usage_data.keys() if k != 'primitive'])
            counts = [usage_data[opt] for opt in options] + [usage_data['primitive']]
            labels = [f'Option {opt}' for opt in options] + ['Primitive Actions']
            
            plt.figure(figsize=(10, 6))
            plt.bar(labels, counts)
            plt.title('Option Usage Frequency')
            plt.ylabel('Selection Count')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.dirs['plots'], f"option_usage{suffix}.png"))
            plt.close()
        
        # 4. State Visitation Heatmap (as in paper Fig. 4)
        if self.exploration_metrics['state_visitation']:
            # Create a 2D grid for the heatmap
            maze_size = self.env.unwrapped.maze_size
            heatmap = np.zeros((maze_size, maze_size))
            
            # Fill in the heatmap with visitation counts
            for (x, y), count in self.exploration_metrics['state_visitation'].items():
                if 0 <= x < maze_size and 0 <= y < maze_size:
                    heatmap[y, x] = count  # Note: y,x for correct orientation
            
            # Plot the heatmap
            plt.figure(figsize=(8, 8))
            sns.heatmap(heatmap, cmap='viridis', cbar=True)
            plt.title('State Visitation Frequency Heatmap')
            plt.savefig(os.path.join(self.dirs['plots'], f"state_visitation{suffix}.png"))
            plt.close()
            
            # Also plot log-scaled version for better visualization of differences
            plt.figure(figsize=(8, 8))
            sns.heatmap(np.log1p(heatmap), cmap='viridis', cbar=True)
            plt.title('State Visitation Frequency (Log Scale)')
            plt.savefig(os.path.join(self.dirs['plots'], f"state_visitation_log{suffix}.png"))
            plt.close()
        
        # 5. Option-specific heatmaps (one per option)
        if any(self.option_metrics['option_state_visitation'].values()):
            for opt_idx in range(self.agent.num_options):
                opt_visitation = self.option_metrics['option_state_visitation'][opt_idx]
                if opt_visitation:
                    # Create heatmap grid
                    maze_size = self.env.unwrapped.maze_size
                    opt_heatmap = np.zeros((maze_size, maze_size))
                    
                    # Fill with visitation counts
                    for (x, y), count in opt_visitation.items():
                        if 0 <= x < maze_size and 0 <= y < maze_size:
                            opt_heatmap[y, x] = count
                    
                    # Plot heatmap
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(opt_heatmap, cmap='viridis', cbar=True)
                    plt.title(f'Option {opt_idx} State Visitation')
                    plt.savefig(os.path.join(self.dirs['plots'], f"option_{opt_idx}_visitation{suffix}.png"))
                    plt.close()
    
    def save_metrics(self):
        """Save all metrics to files following DCEO paper structure."""
        metrics_path = os.path.join(self.results_dir, "metrics.json")
        
        try:
            # Prepare metrics for serialization - organized by paper categories
            metrics_to_save = {
                # 4.5.1 Performance Metrics
                'performance': {
                    'episode_rewards': self.convert_numpy_to_python(self.performance_metrics['episode_rewards']),
                    'episode_lengths': self.convert_numpy_to_python(self.performance_metrics['episode_lengths']),
                    'episode_success': self.convert_numpy_to_python(self.performance_metrics['episode_success']),
                    'episodes_completed': self.performance_metrics['episodes_completed'],
                    'total_env_steps': self.performance_metrics['total_env_steps'],
                    'reward_by_steps': self.convert_numpy_to_python(list(zip(self.performance_metrics['env_steps'], self.performance_metrics['reward_by_steps']))),
                    'success_by_steps': self.convert_numpy_to_python(list(zip(self.performance_metrics['env_steps'], self.performance_metrics['success_by_steps']))) if self.performance_metrics['success_by_steps'] else []
                },
                
                # 4.5.2 Exploration Metrics
                'exploration': {
                    'state_coverage': self.convert_numpy_to_python(self.exploration_metrics['step_coverage']),
                    'unique_states_count': len(self.exploration_metrics['unique_states_visited']),
                    'keys_collected': sum(self.exploration_metrics['keys_collected']),
                    'doors_opened': sum(self.exploration_metrics['door_opened']),
                    'keys_collected_steps': self.convert_numpy_to_python(self.exploration_metrics['keys_collected_steps']),
                    'doors_opened_steps': self.convert_numpy_to_python(self.exploration_metrics['doors_opened_steps']),
                    'goal_reached_steps': self.convert_numpy_to_python(self.exploration_metrics['goal_reached_steps']),
                    'episode_coverage': self.convert_numpy_to_python(self.exploration_metrics['episode_coverage'])
                },
                
                # 4.5.3 Option Quality Metrics
                'option': {
                    'option_selection_counts': self.convert_numpy_to_python(dict(self.option_metrics['option_selection_counts'])),
                    'primitive_action_count': self.option_metrics['primitive_action_count'],
                    'option_durations': {k: self.convert_numpy_to_python(v) for k, v in self.option_metrics['option_durations'].items()},
                    'option_displacements': {k: self.convert_numpy_to_python(v) for k, v in self.option_metrics['option_displacements'].items()},
                    'option_steps': self.convert_numpy_to_python(dict(self.option_metrics['option_steps']))
                },
                
                # Continuous monitoring data
                'continuous': {
                    'step_milestones': self.convert_numpy_to_python(self.continuous_metrics['step_milestones']),
                    'coverage_at_milestones': self.convert_numpy_to_python(self.continuous_metrics['coverage_at_milestones']),
                    'reward_at_milestones': self.convert_numpy_to_python(self.continuous_metrics['reward_at_milestones'])
                },
                
                # Metadata
                'metadata': {
                    'timestamp': self.timestamp,
                    'maze_size': self.env.unwrapped.maze_size,
                    'num_keys': self.env.unwrapped.num_keys,
                    'num_options': self.agent.num_options,
                    'training_time': time.time() - self.start_time
                }
            }
            
            # Save to file
            with open(metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            
            # Save state visitation data separately (can be large)
            np.save(os.path.join(self.dirs['exploration'], "state_visitation.npy"), 
                   dict(self.exploration_metrics['state_visitation']))
            
            # Save option state visitation data
            for opt_idx in range(self.agent.num_options):
                np.save(os.path.join(self.dirs['option'], f"option_{opt_idx}_visitation.npy"),
                       dict(self.option_metrics['option_state_visitation'][opt_idx]))
                
                # Also save option trajectories for behavior analysis
                if self.option_metrics['option_trajectories'][opt_idx]:
                    np.save(os.path.join(self.dirs['option'], f"option_{opt_idx}_trajectories.npy"),
                           self.option_metrics['option_trajectories'][opt_idx])
            
            print(f"Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def generate_plots(self):
        """Generate comprehensive final plots following DCEO paper methodology."""
        # Final suffix
        suffix = "_final"
        
        # Generate DCEO paper style plots
        
        # 1. Learning Curve (Figure 3 in paper)
        if len(self.performance_metrics['env_steps']) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.performance_metrics['env_steps'], self.performance_metrics['reward_by_steps'], 'b-', linewidth=2)
            plt.title('Learning Curve: Average Return vs Environment Steps')
            plt.xlabel('Environment Steps')
            plt.ylabel('Average Return')
            plt.grid(True)
            plt.savefig(os.path.join(self.dirs['plots'], f"learning_curve{suffix}.png"))
            plt.close()
            
            # Also plot success rate if available
            if self.performance_metrics['success_by_steps']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.performance_metrics['env_steps'], self.performance_metrics['success_by_steps'], 'g-', linewidth=2)
                plt.title('Success Rate vs Environment Steps')
                plt.xlabel('Environment Steps')
                plt.ylabel('Success Rate')
                plt.grid(True)
                plt.ylim(0, 1.1)
                plt.savefig(os.path.join(self.dirs['plots'], f"success_rate{suffix}.png"))
                plt.close()
        
        # 2. State Coverage (part of exploration metrics - Figure 4 in paper)
        if len(self.exploration_metrics['step_coverage']) > 1:
            plt.figure(figsize=(10, 6))
            steps, coverages = zip(*self.exploration_metrics['step_coverage'])
            plt.plot(steps, coverages, 'g-', linewidth=2)
            plt.title('State Space Coverage vs Environment Steps')
            plt.xlabel('Environment Steps')
            plt.ylabel('Coverage (%)')
            plt.grid(True)
            plt.savefig(os.path.join(self.dirs['plots'], f"state_coverage{suffix}.png"))
            plt.close()
        
        # 3. State Visitation Heatmap (Figure 4 in paper)
        if self.exploration_metrics['state_visitation']:
            # Create a 2D grid for the heatmap
            maze_size = self.env.unwrapped.maze_size
            heatmap = np.zeros((maze_size, maze_size))
            
            # Fill in the heatmap with visitation counts
            for (x, y), count in self.exploration_metrics['state_visitation'].items():
                if 0 <= x < maze_size and 0 <= y < maze_size:
                    heatmap[y, x] = count
            
            # Plot the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap, cmap='viridis', cbar=True)
            plt.title('State Visitation Frequency Heatmap')
            plt.savefig(os.path.join(self.dirs['plots'], f"state_visitation{suffix}.png"))
            plt.close()
            
            # Also plot log-scaled version for better visualization of differences
            plt.figure(figsize=(10, 8))
            sns.heatmap(np.log1p(heatmap), cmap='viridis', cbar=True)
            plt.title('State Visitation Frequency (Log Scale)')
            plt.savefig(os.path.join(self.dirs['plots'], f"state_visitation_log{suffix}.png"))
            plt.close()
        
        # 4. Option Usage Frequency (related to Figure 5 in paper)
        if self.option_metrics['option_selection_counts']:
            # Prepare data
            opt_ids = sorted(self.option_metrics['option_selection_counts'].keys())
            opt_counts = [self.option_metrics['option_selection_counts'][i] for i in opt_ids]
            
            # Add primitive action count
            labels = [f'Option {i}' for i in opt_ids] + ['Primitive']
            counts = opt_counts + [self.option_metrics['primitive_action_count']]
            
            plt.figure(figsize=(10, 6))
            plt.bar(labels, counts, color='cornflowerblue')
            plt.title('Option Usage Frequency')
            plt.ylabel('Selection Count')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.dirs['plots'], f"option_usage{suffix}.png"))
            plt.close()
            
            # Also plot as percentage
            total = sum(counts)
            if total > 0:  # Avoid division by zero
                percentages = [count/total*100 for count in counts]
                plt.figure(figsize=(10, 6))
                plt.bar(labels, percentages, color='lightseagreen')
                plt.title('Option Usage Percentage')
                plt.ylabel('Usage (%)')
                plt.xticks(rotation=45)
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(self.dirs['plots'], f"option_usage_percent{suffix}.png"))
                plt.close()
        
        # 5. Option-specific behavior analysis (Figure 5 in paper)
        if any(self.option_metrics['option_state_visitation'].values()):
            # First, generate individual option heatmaps
            for opt_idx in range(self.agent.num_options):
                opt_visitation = self.option_metrics['option_state_visitation'][opt_idx]
                if opt_visitation:
                    # Create heatmap grid
                    maze_size = self.env.unwrapped.maze_size
                    opt_heatmap = np.zeros((maze_size, maze_size))
                    
                    # Fill with visitation counts
                    for (x, y), count in opt_visitation.items():
                        if 0 <= x < maze_size and 0 <= y < maze_size:
                            opt_heatmap[y, x] = count
                    
                    # Plot heatmap
                    plt.figure(figsize=(8, 8))
                    sns.heatmap(opt_heatmap, cmap='viridis', cbar=True)
                    plt.title(f'Option {opt_idx} State Visitation')
                    plt.savefig(os.path.join(self.dirs['option'], f"option_{opt_idx}_visitation{suffix}.png"))
                    plt.close()
            
            # Generate option displacement distribution (to analyze behavior)
            if any(self.option_metrics['option_displacements'].values()):
                plt.figure(figsize=(10, 6))
                for opt_idx in range(self.agent.num_options):
                    if opt_idx in self.option_metrics['option_displacements'] and self.option_metrics['option_displacements'][opt_idx]:
                        plt.hist(self.option_metrics['option_displacements'][opt_idx], 
                                alpha=0.5, bins=20, label=f'Option {opt_idx}')
                plt.title('Option Displacement Distribution')
                plt.xlabel('Displacement')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.dirs['option'], f"option_displacement{suffix}.png"))
                plt.close()
            
            # Generate option duration distribution
            if any(self.option_metrics['option_durations'].values()):
                plt.figure(figsize=(10, 6))
                for opt_idx in range(self.agent.num_options):
                    if opt_idx in self.option_metrics['option_durations'] and self.option_metrics['option_durations'][opt_idx]:
                        plt.hist(self.option_metrics['option_durations'][opt_idx], 
                                alpha=0.5, bins=20, label=f'Option {opt_idx}')
                plt.title('Option Duration Distribution')
                plt.xlabel('Duration (steps)')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.dirs['option'], f"option_duration{suffix}.png"))
                plt.close()
    
    def _generate_exploration_plots(self, plots_dir, suffix):
        """Generate exploration metrics plots."""
        # 1. State Coverage Over Time
        if self.exploration_metrics['state_coverage']:
            iterations, coverages = zip(*self.exploration_metrics['state_coverage'])
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, coverages, 'b-', linewidth=2)
            plt.title('State Space Coverage Over Time')
            plt.xlabel('Iteration')
            plt.ylabel('Coverage (%)')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"state_coverage{suffix}.png"))
            plt.close()
        
        # 2. State Visitation Heatmap
        if self.exploration_metrics['state_visitation']:
            # Create a 2D grid for the heatmap
            maze_size = self.env.unwrapped.maze_size
            heatmap = np.zeros((maze_size, maze_size))
            
            # Fill in the heatmap with visitation counts
            max_count = 1  # To avoid division by zero
            for (x, y), count in self.exploration_metrics['state_visitation'].items():
                if 0 <= x < maze_size and 0 <= y < maze_size:
                    heatmap[y, x] = count
                    max_count = max(max_count, count)
            
            # Plot the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap, cmap='viridis', vmin=0, vmax=max_count)
            plt.title('State Visitation Frequency Heatmap')
            plt.savefig(os.path.join(plots_dir, f"state_visitation_heatmap{suffix}.png"))
            plt.close()
            
            # Also save a log-scale version for better visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(np.log1p(heatmap), cmap='viridis')  # log1p for better visualization
            plt.title('State Visitation Frequency Heatmap (Log Scale)')
            plt.savefig(os.path.join(plots_dir, f"state_visitation_heatmap_log{suffix}.png"))
            plt.close()
        
        # 3. Key Collection and Door Opening Statistics
        key_steps = self.exploration_metrics['key_collection_steps']
        door_steps = self.exploration_metrics['door_opening_steps']
        goal_steps = self.exploration_metrics['goal_reaching_steps']
        
        if key_steps or door_steps or goal_steps:
            plt.figure(figsize=(12, 6))
            
            if key_steps:
                plt.hist(key_steps, bins=20, alpha=0.5, label='Key Collection')
            
            if door_steps:
                plt.hist(door_steps, bins=20, alpha=0.5, label='Door Opening')
                
            if goal_steps:
                plt.hist(goal_steps, bins=20, alpha=0.5, label='Goal Reaching')
            
            plt.title('Distribution of Steps to Key Events')
            plt.xlabel('Episode Steps')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"key_events_distribution{suffix}.png"))
            plt.close()
    
    def _generate_option_quality_plots(self, plots_dir, suffix):
        """Generate option quality metrics plots."""
        # 1. Option Usage Frequency
        if self.option_metrics['option_usage_counts']:
            # Convert to list for plotting
            option_ids = sorted(self.option_metrics['option_usage_counts'].keys())
            usage_counts = [self.option_metrics['option_usage_counts'][i] for i in option_ids]
            
            plt.figure(figsize=(10, 6))
            plt.bar(option_ids, usage_counts, color='green')
            plt.title('Option Usage Frequency')
            plt.xlabel('Option ID')
            plt.ylabel('Usage Count')
            plt.xticks(option_ids)
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(plots_dir, f"option_usage_frequency{suffix}.png"))
            plt.close()
        
        # 2. Option Duration Distribution
        if any(self.option_metrics['option_durations'].values()):
            plt.figure(figsize=(12, 6))
            
            for opt_id, durations in self.option_metrics['option_durations'].items():
                if durations:
                    plt.hist(durations, bins=20, alpha=0.5, label=f'Option {opt_id}')
            
            plt.title('Option Duration Distribution')
            plt.xlabel('Duration (steps)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"option_duration_distribution{suffix}.png"))
            plt.close()
        
        # 3. Option Reward Distribution
        if any(self.option_metrics['option_rewards'].values()):
            plt.figure(figsize=(12, 6))
            
            for opt_id, rewards in self.option_metrics['option_rewards'].items():
                if rewards:
                    plt.hist(rewards, bins=20, alpha=0.5, label=f'Option {opt_id}')
            
            plt.title('Option Reward Distribution')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"option_reward_distribution{suffix}.png"))
            plt.close()
        
        # 4. Option Usage Over Time
        if self.option_time_metrics['steps']:
            steps = self.option_time_metrics['steps']
            usage_data = np.array(self.option_time_metrics['option_usage'])
            
            plt.figure(figsize=(12, 6))
            
            for i in range(self.agent.num_options):
                plt.plot(steps, usage_data[:, i], label=f'Option {i}')
            
            plt.title('Option Usage Over Time')
            plt.xlabel('Training Steps')
            plt.ylabel('Usage Count')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"option_usage_over_time{suffix}.png"))
            plt.close()
        
        # 5. Option State Visitation Heatmaps (one per option)
        maze_size = self.env.unwrapped.maze_size
        
        for opt_id in range(self.agent.num_options):
            if self.option_metrics['option_state_visitation'][opt_id]:
                # Create a 2D grid for the heatmap
                heatmap = np.zeros((maze_size, maze_size))
                
                # Fill in the heatmap with visitation counts
                max_count = 1  # To avoid division by zero
                for (x, y), count in self.option_metrics['option_state_visitation'][opt_id].items():
                    if 0 <= x < maze_size and 0 <= y < maze_size:
                        heatmap[y, x] = count
                        max_count = max(max_count, count)
                
                # Plot the heatmap
                plt.figure(figsize=(8, 6))
                sns.heatmap(heatmap, cmap='viridis', vmin=0, vmax=max_count)
                plt.title(f'Option {opt_id} State Visitation Heatmap')
                plt.savefig(os.path.join(plots_dir, f"option_{opt_id}_visitation{suffix}.png"))
                plt.close()
    
    def _generate_performance_plots(self, plots_dir, suffix):
        """Generate performance metrics plots."""
        # 1. Episode Rewards
        if self.performance_metrics['episode_rewards']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.performance_metrics['episode_rewards'], 'r-')
            plt.title('Episode Rewards During Training')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"episode_rewards{suffix}.png"))
            plt.close()
        
        # 2. Episode Lengths
        if self.performance_metrics['episode_lengths']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.performance_metrics['episode_lengths'], 'b-')
            plt.title('Episode Lengths During Training')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"episode_lengths{suffix}.png"))
            plt.close()
        
        # 3. Evaluation Metrics
        if self.performance_metrics['evaluation_rewards']:
            iterations, rewards = zip(*self.performance_metrics['evaluation_rewards'])
            iterations2, success_rates = zip(*self.performance_metrics['evaluation_success_rates'])
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, rewards, 'g-', label='Avg Reward')
            plt.plot(iterations2, success_rates, 'b-', label='Success Rate')
            plt.title('Evaluation Metrics')
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"evaluation_metrics{suffix}.png"))
            plt.close()
    
    def _extract_agent_position(self, state):
        """Extract agent position from state representation.
        
        This is a fallback method if the agent's position isn't directly accessible.
        
        Args:
            state: State observation
            
        Returns:
            Tuple (x, y) of agent position, or None if not extractable
        """
        try:
            if isinstance(state, tuple) and len(state) > 0:
                state = state[0]  # Handle (state, info) format
                
            if isinstance(state, np.ndarray):
                # For multi-channel grid representation (channel 0 is agent)
                if len(state.shape) == 3 and state.shape[2] >= 5:
                    agent_channel = state[:, :, 0]
                    agent_pos = np.where(agent_channel > 0)
                    if len(agent_pos[0]) > 0 and len(agent_pos[1]) > 0:
                        return (int(agent_pos[1][0]), int(agent_pos[0][0]))
        except Exception:
            pass
            
        return None
    
    def convert_numpy_to_python(self, obj):
        """Convert numpy types to standard Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [self.convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_to_python(v) for k, v in obj.items()}
        else:
            return obj
