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
import torch
import random
import matplotlib.pyplot as plt
from collections import deque
import argparse
import time
import json

# Add parent directory to path to allow imports from maze_experiment
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'maze_experiment'))

# Import original agent implementations
from maze_experiment.standard_q_agent import StandardDQNAgent
from maze_experiment.count_based_agent import CountBasedAgent
from maze_experiment.rnd_agent import RNDAgent
from key_maze_dceo import KeyMazeDCEOAgent

# Import Key-Door Maze environment and wrapper
from key_door_maze import KeyDoorMaze
from key_maze_wrapper import KeyMazeWrapper

# Set random seeds for reproducibility
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare agents on Key-Door Maze environment')
parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train each agent')
parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate after this many episodes')
parser.add_argument('--maze_size', type=int, default=10, help='Size of the maze')
parser.add_argument('--max_steps', type=int, default=200, help='Maximum steps per episode')
parser.add_argument('--num_keys', type=int, default=2, help='Number of keys in the maze')
parser.add_argument('--skip', type=str, nargs='*', default=[], help='Agents to skip')
parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
parser.add_argument('--fixed_maze', action='store_true', help='Use fixed maze layout across episodes')
parser.add_argument('--maze_seed', type=int, default=42, help='Random seed for maze generation')
parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility')
parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon for exploration')
parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final epsilon for exploration')
parser.add_argument('--epsilon_decay', type=float, default=0.99, help='Epsilon decay rate per episode')
parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
args = parser.parse_args()

# Create the results directory
os.makedirs(args.results_dir, exist_ok=True)

def evaluate_agent(env, agent, num_episodes):
    """Evaluate agent performance over multiple episodes"""
    rewards = []
    successes = 0
    total_steps = 0
    total_keys = 0
    total_doors = 0
    total_coverage = 0
    
    for episode in range(num_episodes):
        # Always use the same maze for evaluation for consistency
        state = env.reset(seed=args.maze_seed if args.fixed_maze else None)
        state = np.transpose(state, (2, 0, 1))  # NHWC to NCHW
        
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        keys_found = 0
        doors_opened = 0
        
        while not done and not truncated and steps < args.max_steps:
            # Select action without exploration
            action = agent.select_action(state, epsilon=0.0, eval_mode=True)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            
            # Convert to PyTorch format
            next_state = np.transpose(next_state, (2, 0, 1))
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Track key collection and door opening
            if reward >= 0.5 and reward < 1.0:  # Key collected
                keys_found += 1
            elif reward >= 1.0 and reward < 10.0:  # Door opened
                doors_opened += 1
            
            # Check if goal reached (more explicit tracking)
            if reward >= 10.0:
                print(f"Goal reached in evaluation! Steps: {steps}")
        
        # Check if successful (reached goal)
        if done and not truncated:
            successes += 1
        
        rewards.append(episode_reward)
        total_steps += steps
        total_keys += keys_found
        total_doors += doors_opened
        total_coverage += info.get("coverage", 0)
    
    return {
        "reward": np.mean(rewards),
        "success_rate": successes / num_episodes,
        "steps": total_steps / num_episodes,
        "keys": total_keys / num_episodes,
        "doors": total_doors / num_episodes,
        "coverage": total_coverage / num_episodes
    }

def get_epsilon(episode):
    """Get epsilon value for episode"""
    # Exponential decay
    return max(args.epsilon_start * (args.epsilon_decay ** episode), args.epsilon_end)

def train_and_evaluate(agent, env, agent_name, results_dir, color):
    """Train and evaluate an agent, return metrics for comparison."""
    
    # Training metrics
    rewards = []
    success_rates = []
    episode_steps = []
    eval_rewards = []
    eval_success_rates = []
    eval_steps = []
    eval_keys = []
    eval_doors = []
    eval_coverage = []
    
    # Create agent-specific results directory
    agent_dir = os.path.join(results_dir, agent_name)
    os.makedirs(agent_dir, exist_ok=True)
    
    # Initial evaluation
    eval_results = evaluate_agent(env, agent, args.eval_episodes)
    eval_rewards.append(eval_results["reward"])
    eval_success_rates.append(eval_results["success_rate"])
    eval_steps.append(eval_results["steps"])
    eval_keys.append(eval_results["keys"])
    eval_doors.append(eval_results["doors"])
    eval_coverage.append(eval_results["coverage"])
    
    print(f"\n----- Initial {agent_name} Evaluation -----")
    print(f"Success Rate: {eval_results['success_rate']:.4f}, Avg Reward: {eval_results['reward']:.4f}")
    print(f"Avg Steps: {eval_results['steps']:.2f}, Avg Keys: {eval_results['keys']:.2f}, Avg Doors: {eval_results['doors']:.2f}")
    print(f"Avg Coverage: {eval_results['coverage']:.2f}%")
    
    start_time = time.time()
    
    # Training loop
    for episode in range(args.episodes):
        episode_start = time.time()
        print(f"\n======= {agent_name}: Episode {episode+1}/{args.episodes} =======\n")
        
        state = env.reset()
        # Convert to PyTorch format (NCHW)
        state = np.transpose(state, (2, 0, 1))  # NHWC to NCHW
        
        epsilon = get_epsilon(episode)
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        keys_found = 0
        doors_opened = 0
        
        while not done and not truncated and steps < args.max_steps:
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            
            # Convert to PyTorch format
            next_state = np.transpose(next_state, (2, 0, 1))
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if hasattr(agent, 'batch_size') and len(agent.memory) >= agent.batch_size:
                loss_info = agent.train()
                if steps % 50 == 0 and loss_info:
                    if isinstance(loss_info, dict):
                        print(f"  [Step {steps}] Losses: " + 
                              ", ".join([f"{k}={v:.4f}" for k, v in loss_info.items()]))
                    else:
                        print(f"  [Step {steps}] Loss: {loss_info:.4f}")
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Track key collection and door opening
            if reward >= 0.5 and reward < 1.0:  # Key collected
                keys_found += 1
            elif reward >= 1.0 and reward < 10.0:  # Door opened
                doors_opened += 1
        
        # Record episode metrics
        rewards.append(episode_reward)
        episode_steps.append(steps)
        
        # Print episode results
        episode_time = time.time() - episode_start
        print(f"{agent_name} Episode {episode+1} completed in {episode_time:.2f}s:")
        print(f"  Steps: {steps}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Epsilon: {epsilon:.4f}")
        print(f"  Keys found: {keys_found}")
        print(f"  Doors opened: {doors_opened}")
        print(f"  Buffer size: {len(agent.memory)}")
        print(f"  Coverage: {info.get('coverage', 'N/A')}%")
        
        # Evaluate agent
        if (episode + 1) % args.eval_interval == 0:
            print(f"\n----- {agent_name} Evaluation after episode {episode+1} -----")
            eval_results = evaluate_agent(env, agent, args.eval_episodes)
            
            eval_rewards.append(eval_results["reward"])
            eval_success_rates.append(eval_results["success_rate"])
            eval_steps.append(eval_results["steps"])
            eval_keys.append(eval_results["keys"])
            eval_doors.append(eval_results["doors"])
            eval_coverage.append(eval_results["coverage"])
            
            print(f"Success Rate: {eval_results['success_rate']:.4f} ({int(eval_results['success_rate'] * args.eval_episodes)}/{args.eval_episodes})")
            print(f"Avg Reward: {eval_results['reward']:.4f}")
            print(f"Avg Steps: {eval_results['steps']:.2f}")
            print(f"Avg Keys: {eval_results['keys']:.2f}")
            print(f"Avg Doors: {eval_results['doors']:.2f}")
            print(f"Avg Coverage: {eval_results['coverage']:.2f}%\n")
    
    total_time = time.time() - start_time
    print(f"\n{agent_name} training completed in {total_time:.2f}s")
    
    # Save metrics
    metrics = {
        'rewards': rewards,
        'episode_steps': episode_steps,
        'eval_rewards': eval_rewards,
        'eval_success_rates': eval_success_rates,
        'eval_steps': eval_steps,
        'eval_keys': eval_keys,
        'eval_doors': eval_doors,
        'eval_coverage': eval_coverage,
        'training_time': total_time
    }
    
    with open(os.path.join(agent_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    
    # Plot success rates
    eval_episodes = list(range(0, args.episodes + 1, args.eval_interval))
    if len(eval_episodes) != len(eval_success_rates):
        eval_episodes = eval_episodes[:len(eval_success_rates)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(eval_episodes, eval_success_rates, marker='o', color=color, label=agent_name)
    plt.title(f'{agent_name} Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(agent_dir, 'success_rate.png'))
    plt.close()
    
    return {
        'name': agent_name,
        'color': color,
        'episodes': eval_episodes,
        'success_rates': eval_success_rates,
        'rewards': eval_rewards,
        'steps': eval_steps,
        'keys': eval_keys,
        'doors': eval_doors,
        'coverage': eval_coverage,
        'training_time': total_time
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
    env = KeyMazeWrapper(
        KeyDoorMaze(size=args.maze_size, num_keys=args.num_keys, max_steps=args.max_steps),
        frame_stack=4,
        resize_shape=(84, 84)
    )

    # Set fixed maze seed if specified
    if args.fixed_maze:
        print(f"Using fixed maze layout with seed {args.maze_seed}")
    
    # Get observation shape and convert to PyTorch format (NCHW)
    obs_shape = env.observation_space.shape  # (H, W, C)
    pytorch_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
    action_size = env.action_space.n
    
    print(f"Environment: Key-Door Maze")
    print(f"Observation shape: {obs_shape}, PyTorch shape: {pytorch_shape}")
    print(f"Action size: {action_size}")
    
    results = []
    
    # Define the agents to train and evaluate
    if 'standard' not in args.skip:
        print("\n===== Training Standard DQN Agent =====")
        from key_maze_agents import KeyMazeStandardAgent
        standard_agent = KeyMazeStandardAgent(input_shape=(4, 84, 84), num_actions=4, buffer_size=args.buffer_size)
        standard_agent.to(device)
        standard_results = train_and_evaluate(standard_agent, env, "StandardDQN", args.results_dir, 'blue')
        results.append(standard_results)
    
    if 'count' not in args.skip:
        print("\n===== Training Count-Based Exploration Agent =====")
        from key_maze_agents import KeyMazeCountAgent
        count_agent = KeyMazeCountAgent(input_shape=(4, 84, 84), num_actions=4, buffer_size=args.buffer_size)
        count_agent.to(device)
        count_results = train_and_evaluate(count_agent, env, "CountBased", args.results_dir, 'green')
        results.append(count_results)
    
    if 'rnd' not in args.skip:
        print("\n===== Training RND Agent =====")
        from key_maze_agents import KeyMazeRNDAgent
        rnd_agent = KeyMazeRNDAgent(input_shape=(4, 84, 84), num_actions=4, buffer_size=args.buffer_size)
        rnd_agent.to(device)
        rnd_results = train_and_evaluate(rnd_agent, env, "RND", args.results_dir, 'orange')
        results.append(rnd_results)
        
    if 'dceo' not in args.skip:
        print("\n===== Training Rainbow DCEO Agent =====\n")
        from key_maze_dceo import KeyMazeDCEOAgent
        from key_maze_networks import KeyMazeDQNNetwork, KeyMazeRainbowNetwork, KeyMazeRepNetwork
        dceo_agent = KeyMazeDCEOAgent(
            state_shape=(4, 84, 84),
            action_size=4,
            network=KeyMazeRainbowNetwork,
            rep_network=KeyMazeRepNetwork,
            buffer_size=args.buffer_size,
            option_prob=0.8,  # Increase option usage probability
            option_duration=15,  # Longer option duration for key-door navigation
            prioritized_replay=True,
            noisy_nets=True,
            dueling=True,
            double_dqn=True
        )
        dceo_agent.to(device)
        dceo_results = train_and_evaluate(dceo_agent, env, "RainbowDCEO", args.results_dir, 'red')
        results.append(dceo_results)
    
    # Create comparison plots
    create_comparison_plots(results)

def create_comparison_plots(results):
    """Create plots comparing all agents."""
    if not results:
        print("No results to plot!")
        return
        
    plt.figure(figsize=(15, 12))
    
    # Success Rate Comparison
    plt.subplot(3, 2, 1)
    for result in results:
        plt.plot(result['episodes'], result['success_rates'], marker='o', 
                 color=result['color'], label=result['name'])
    plt.title('Success Rate Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Reward Comparison
    plt.subplot(3, 2, 2)
    for result in results:
        episodes = result['episodes']
        if len(episodes) > 0:  # Skip empty results
            plt.plot(episodes, result['rewards'], marker='o', 
                    color=result['color'], label=result['name'])
    plt.title('Average Reward Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Steps Comparison
    plt.subplot(3, 2, 3)
    for result in results:
        episodes = result['episodes']
        if len(episodes) > 0:  # Skip empty results
            plt.plot(episodes, result['steps'], marker='o', 
                    color=result['color'], label=result['name'])
    plt.title('Average Steps Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Keys Collected Comparison
    plt.subplot(3, 2, 4)
    for result in results:
        episodes = result['episodes']
        if len(episodes) > 0 and 'keys' in result:  # Skip empty results
            plt.plot(episodes, result['keys'], marker='o', 
                    color=result['color'], label=result['name'])
    plt.title('Average Keys Collected')
    plt.xlabel('Episode')
    plt.ylabel('Keys Collected')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Doors Opened Comparison
    plt.subplot(3, 2, 5)
    for result in results:
        episodes = result['episodes']
        if len(episodes) > 0 and 'doors' in result:  # Skip empty results
            plt.plot(episodes, result['doors'], marker='o', 
                    color=result['color'], label=result['name'])
    plt.title('Average Doors Opened')
    plt.xlabel('Episode')
    plt.ylabel('Doors Opened')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Training Time Comparison
    plt.subplot(3, 2, 6)
    agent_names = [result['name'] for result in results]
    train_times = [result['training_time'] for result in results]
    colors = [result['color'] for result in results]
    
    plt.bar(agent_names, train_times, color=colors)
    plt.title('Training Time Comparison')
    plt.xlabel('Agent')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'agent_comparison.png'))
    plt.close()
    
    # Create additional plot for coverage comparison
    plt.figure(figsize=(10, 6))
    for result in results:
        episodes = result['episodes']
        if len(episodes) > 0 and 'coverage' in result:  # Skip empty results
            plt.plot(episodes, result['coverage'], marker='o', 
                    color=result['color'], label=result['name'])
    plt.title('Environment Coverage Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Coverage (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.results_dir, 'coverage_comparison.png'))
    plt.close()
    
    print(f"Comparison results saved to {args.results_dir}")

if __name__ == "__main__":
    compare_all_agents()
