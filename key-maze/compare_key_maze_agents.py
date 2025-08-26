"""
Compare the performance of Rainbow DCEO with baseline agents in the Key-Door Maze environment:
1. Standard Q-learning (DQN)
2. Count-based exploration
3. Random Network Distillation (RND)

This script trains all agents on the Key-Door Maze environment and plots their performance.
The Key-Door Maze requires hierarchical exploration, making it ideal for testing option discovery.
"""

import os
import sys
import numpy as np
import random
import torch
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import time
import json
from tqdm import tqdm
from IPython.display import clear_output

# Add parent directory to path to allow imports from maze_experiment
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'maze_experiment'))

# Import environment and agents
from key_door_maze_env import KeyDoorMazeEnv
from dceo_key_maze_wrapper import DCEOKeyMazeWrapper
from key_maze_dceo_agent import KeyMazeDCEOAgent
from maze_experiment.standard_q_agent import StandardDQNAgent
from maze_experiment.count_based_agent import CountBasedAgent
from maze_experiment.rnd_agent import RNDAgent

# Set random seeds for reproducibility
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare agents on Key-Door Maze environment')
parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train each agent')
parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
parser.add_argument('--eval_interval', type=int, default=50, help='Evaluate after this many episodes')
parser.add_argument('--maze_size', type=int, default=10, help='Size of the maze')
parser.add_argument('--max_steps', type=int, default=200, help='Maximum steps per episode')
parser.add_argument('--num_keys', type=int, default=2, help='Number of keys in the maze')
parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
parser.add_argument('--fixed_maze', action='store_true', help='Use fixed maze layout across episodes')
parser.add_argument('--maze_seed', type=int, default=42, help='Random seed for maze generation')
parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility')
parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon for exploration')
parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final epsilon for exploration')
parser.add_argument('--epsilon_decay', type=float, default=0.99, help='Epsilon decay rate per episode')
parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
parser.add_argument('--skip', type=str, nargs='+', default=[], choices=['standard', 'count', 'rnd', 'dceo'], 
                    help='Skip specified agents')
parser.add_argument('--plot_results', action='store_true', help='Plot training results')
args = parser.parse_args()

# Create results directory if it doesn't exist
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate an agent on the environment without exploration."""
    total_reward = 0
    success_count = 0
    key_count = 0
    door_count = 0
    coverage_total = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select action without exploration
            action = agent.select_action(obs, epsilon=0.001, eval_mode=True)
            
            # Take action in environment - handle both gym interface versions
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                
            episode_reward += reward
        
        # Check if goal was reached
        if info.get('goal_reached', False):
            success_count += 1
            
        # Count keys collected and doors opened
        key_count += info.get('keys_collected', 0)
        door_count += info.get('doors_opened', 0)
        
        # Track coverage
        coverage_stats = env.get_coverage_stats()
        coverage_total += coverage_stats['coverage']
        
        total_reward += episode_reward
    
    # Calculate average metrics
    avg_reward = total_reward / num_episodes
    success_rate = success_count / num_episodes
    avg_keys = key_count / num_episodes
    avg_doors = door_count / num_episodes
    avg_coverage = coverage_total / num_episodes
    
    return {
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'avg_keys': avg_keys,
        'avg_doors': avg_doors,
        'avg_coverage': avg_coverage
    }

def train_and_evaluate(agent, env, agent_name, results_dir=None, color='red', eval_interval=10, num_eval_episodes=10):
    """Train and evaluate an agent on the Key-Door Maze environment.
    
    Args:
        agent: The agent to train and evaluate
        env: The environment to train on
        agent_name: Name of the agent for reporting
        results_dir: Directory to save results
        color: Color to use for plotting agent results
        eval_interval: Number of episodes between evaluations
        num_eval_episodes: Number of episodes to evaluate on
        
    Returns:
        Dictionary of training and evaluation results
    """
    # Initialize tracking variables
    rewards = []
    eval_rewards = []
    eval_success_rates = []
    eval_key_rates = []
    eval_door_rates = []
    eval_coverages = []
    eval_episodes = []
    
    # Setup real-time visualization
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))
    
    start_time = time.time()
    
    # Training loop
    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        keys_collected = 0
        doors_opened = 0
        option_changes = 0
        last_option = None
        epsilon = getattr(agent, 'epsilon', 0.0)  # Get epsilon if it exists
        
        print(f"Starting Episode {episode} with epsilon {epsilon:.4f}")
        
        # Get base environment for visualization
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
            
        reward_pattern = []
        
        while not done and steps < args.max_steps:
            steps += 1
            
            # Get current option information if this is an option-based agent
            if hasattr(agent, 'current_option'):
                current_option = agent.current_option
                if last_option is not None and current_option != last_option:
                    option_changes += 1
                    print(f"  Step {steps}: Switched to option {current_option}")
                last_option = current_option
            
            # Select action based on current observation
            action = agent.act(obs)
            # Handle both gym interface versions (4 or 5 return values)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
            
            # Track rewards
            episode_reward += reward
            reward_pattern.append(round(reward, 1))
            if len(reward_pattern) > 5:
                reward_pattern.pop(0)
            
            # Store transition for learning
            agent.remember(obs, action, reward, next_obs, done)
            
            # Track environment achievements
            if info.get('key_collected', False):
                keys_collected += 1
                print(f"  Step {steps}: Collected key {keys_collected}")
            if info.get('door_opened', False):
                doors_opened += 1
                print(f"  Step {steps}: Opened door {doors_opened}")
            
            # Update observation
            obs = next_obs
            
            # Train the agent
            agent.train()
            
            # Update epsilon if it exists
            epsilon = getattr(agent, 'epsilon', 0.0)
            
            # Show status every 50 steps
            if steps % 50 == 0:
                print(f"  Step {steps}: Reward so far: {episode_reward:.2f}, Keys: {keys_collected}, Doors: {doors_opened}")
                print(f"  Current epsilon: {epsilon:.4f}")
                if hasattr(agent, 'current_option'):
                    # Just display the current option without calculating duration
                    print(f"  Current option: {agent.current_option}")
                    
            # Visualize the environment state every few steps or on key events
            if steps % 5 == 0 or info.get('key_collected', False) or info.get('door_opened', False) or done:
                # Clear the axis and render the environment
                ax.clear()
                maze_img = base_env.render(mode='rgb_array')
                if maze_img is not None:
                    ax.imshow(maze_img)
                    
                    # Display agent information
                    title = f"Episode {episode}, Step {steps}, Reward: {episode_reward:.2f}\n"
                    title += f"Keys: {keys_collected}/{env.unwrapped.num_keys if hasattr(env.unwrapped, 'num_keys') else '?'}, "
                    title += f"Doors: {doors_opened}\n"
                    if hasattr(agent, 'current_option'):
                        title += f"Current Option: {agent.current_option}, Epsilon: {epsilon:.4f}"
                    ax.set_title(title)
                    
                    # Show the plot
                    plt.draw()
                    plt.pause(0.01)  # Short pause to update the plot
        
        # Episode summary
        print(f"Episode {episode} done in {steps} steps")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Keys collected: {keys_collected}/{env.unwrapped.num_keys}")
        print(f"  Doors opened: {doors_opened}")
        print(f"  Goal reached: {info.get('goal_reached', False)}")
        print(f"  Option changes: {option_changes}")
        print(f"  Reward pattern: {', '.join([str(r) for r in reward_pattern])}... (last 5 steps)")
        if hasattr(agent, 'epsilon'):
            print(f"  Final epsilon: {epsilon:.4f}")
        
        # Store training results
        rewards.append(episode_reward)
        
        # Evaluate periodically
        if episode % args.eval_interval == 0:
            print(f"\nEvaluating after episode {episode}...")
            eval_results = evaluate_agent(agent, env, num_episodes=args.eval_episodes)
            print(f"  Eval reward: {eval_results['avg_reward']:.2f}")
            print(f"  Success rate: {eval_results['success_rate']:.2f}")
            print(f"  Keys collected: {eval_results['avg_keys']:.2f}/{env.unwrapped.num_keys}")
            print(f"  Doors opened: {eval_results['avg_doors']:.2f}")
            print(f"  Coverage: {eval_results['avg_coverage']:.2f}\n")
            
            # Store evaluation results
            eval_rewards.append(eval_results['avg_reward'])
            eval_success_rates.append(eval_results['success_rate'])
            eval_key_rates.append(eval_results['avg_keys'])
            eval_door_rates.append(eval_results['avg_doors'])
            eval_coverages.append(eval_results['avg_coverage'])
            eval_episodes.append(episode)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Return results dictionary
    return {
        'agent_name': agent_name,
        'train_rewards': rewards,
        'eval_rewards': eval_rewards,
        'eval_success_rates': eval_success_rates,
        'eval_key_rates': eval_key_rates,
        'eval_door_rates': eval_door_rates,
        'eval_coverages': eval_coverages,
        'eval_episodes': eval_episodes
    }

def compare_all_agents():
    """Train and compare all agents on the Key-Door Maze environment."""
    print("Starting agent comparison experiment with Key-Door Maze")
    print(f"Training each agent for {args.episodes} episodes")
    print(f"Evaluating every {args.eval_interval} episodes with {args.eval_episodes} eval episodes")
    print(f"Maze size: {args.maze_size}, Keys: {args.num_keys}, Max steps: {args.max_steps}")
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Configure device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    maze_seed = args.maze_seed if args.fixed_maze else None
    env = KeyDoorMazeEnv(
        maze_size=args.maze_size, 
        num_keys=args.num_keys, 
        max_steps=args.max_steps,
        use_fixed_layout=args.fixed_maze,
        use_fixed_seed=args.fixed_maze,
        fixed_seed=maze_seed
    )
    
    # Wrap environment for agent compatibility
    # Use 84x84 shape which is standard for DQN-style networks
    env = DCEOKeyMazeWrapper(
        env,
        frame_stack=4,
        resize_shape=(84, 84),
        proximity_reward=True
    )
    
    # Set fixed maze seed if specified
    if args.fixed_maze:
        print(f"Using fixed maze layout with seed {args.maze_seed}")
    
    # Our observation is already in PyTorch format (NCHW) from the wrapper
    # The wrapper returns (C, H, W) format directly
    obs_shape = env.observation_space.shape  # (C, H, W)
    pytorch_shape = obs_shape  # Already in the correct format
    
    print(f"Environment: Key-Door Maze")
    print(f"Observation shape: {obs_shape}, PyTorch shape: {pytorch_shape}")
    print(f"Action size: {env.action_space.n}")
    
    # Store results for each agent
    results = []
    
    # Train and evaluate Standard DQN agent
    if 'standard' not in args.skip:
        print("\n===== Training Standard DQN Agent =====")
        standard_agent = StandardDQNAgent(
            state_shape=pytorch_shape,
            action_size=env.action_space.n,
            buffer_size=args.buffer_size
        )
        standard_agent.to(device)
        standard_results = train_and_evaluate(standard_agent, env, "StandardDQN", args.results_dir, 'blue')
        results.append(standard_results)
    
    # Train and evaluate Count-based exploration agent
    if 'count' not in args.skip:
        print("\n===== Training Count-Based Exploration Agent =====")
        count_agent = CountBasedAgent(
            state_shape=pytorch_shape,
            action_size=env.action_space.n,
            buffer_size=args.buffer_size
        )
        count_agent.to(device)
        count_results = train_and_evaluate(count_agent, env, "CountBased", args.results_dir, 'green')
        results.append(count_results)
    
    # Train and evaluate RND agent
    if 'rnd' not in args.skip:
        print("\n===== Training RND Agent =====")
        rnd_agent = RNDAgent(
            state_shape=pytorch_shape,
            action_size=env.action_space.n,
            buffer_size=args.buffer_size
        )
        rnd_agent.to(device)
        rnd_results = train_and_evaluate(rnd_agent, env, "RND", args.results_dir, 'orange')
        results.append(rnd_results)
        
    # Train and evaluate Rainbow DCEO agent
    if 'dceo' not in args.skip:
        print("\n===== Training Rainbow DCEO Agent =====")
        # Match hyperparameters from the DCEO paper and JAX implementation
        num_options = 8       # Paper uses 8 eigenoptions
        rep_dim = 8           # Representation dimension should match number of options
        gamma = 0.99          # Discount factor from the paper
        learning_rate = 1e-4  # Learning rate from the paper (0.0001)
        buffer_size = 100000  # Larger buffer size as in the paper
        
        dceo_agent = KeyMazeDCEOAgent(
            state_shape=pytorch_shape,
            action_size=env.action_space.n,
            buffer_size=buffer_size,    # Use larger buffer from paper
            learning_rate=learning_rate, # Paper-specific learning rate
            gamma=gamma,                # Paper-specific discount factor
            num_options=num_options,    # 8 eigenoptions as in the paper
            rep_dim=rep_dim,            # 8 dimensions for representation
            option_prob=0.8,            # Higher probability to use options
            option_duration=15,         # Longer option duration for key-door navigation
            prioritized_replay=True,    # Paper uses prioritized replay
            noisy_nets=True,            # Paper uses noisy networks
            dueling=True,               # Paper uses dueling networks
            double_dqn=True,            # Paper uses double DQN
            eps_start=args.epsilon_start,
            eps_end=args.epsilon_end,
            eps_decay=args.epsilon_decay
        )
        # Move agent to device
        dceo_agent.to(device)
        
        # Note: We're implementing the paper's approach with hyperparameters that match
        # but we'll skip the pre-training phase for now to get the agent working
        
        # Train and evaluate agent
        dceo_results = train_and_evaluate(dceo_agent, env, "RainbowDCEO", args.results_dir, 'red')
        results.append(dceo_results)
    
    # Create comparison plots
    create_comparison_plots(results)

def create_comparison_plots(results):
    """Create plots comparing all agents."""
    if not results:
        print("No results to plot!")
        return
    
    # Turn off interactive mode for the final plots
    plt.ioff()
    plt.close()
    
    # Plot training progress
    if args.plot_results and results:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        for result in results:
            plt.plot(result['eval_episodes'], result['eval_rewards'], marker='o', linestyle='-', label=result['agent_name'])
        plt.title('Evaluation Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        for result in results:
            plt.plot(result['eval_episodes'], result['eval_success_rates'], marker='o', linestyle='-', label=result['agent_name'])
        plt.title('Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        for result in results:
            plt.plot(result['eval_episodes'], result['eval_key_rates'], marker='o', linestyle='-', label='Keys')
            plt.plot(result['eval_episodes'], result['eval_door_rates'], marker='s', linestyle='-', label='Doors')
        plt.title('Keys and Doors')
        plt.xlabel('Episode')
        plt.ylabel('Average Number')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        for result in results:
            plt.plot(result['eval_episodes'], result['eval_coverages'], marker='o', linestyle='-', label='Coverage')
        plt.title('Maze Coverage')
        plt.xlabel('Episode')
        plt.ylabel('Coverage Ratio')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot if results directory is provided
        if args.results_dir:
            os.makedirs(args.results_dir, exist_ok=True)
            plt.savefig(os.path.join(args.results_dir, 'agent_comparison.png'))
        
        plt.show()
    # Training Time Comparison
    plt.subplot(3, 2, 6)
    agent_names = [result['agent'] for result in results]
    training_times = [result['training_time'] for result in results]
    colors = [result['color'] for result in results]
    
    plt.bar(agent_names, training_times, color=colors)
    plt.title('Training Time Comparison')
    plt.xlabel('Agent')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'agent_comparison.png'))
    plt.show()

if __name__ == "__main__":
    compare_all_agents()
