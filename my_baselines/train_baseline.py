"""
Train baseline agents on maze environments.
Supports standard Q-Learning, DDQN with count-based exploration, and RND.
"""

import argparse
import numpy as np
import torch
import os
import time
import sys
from tqdm import tqdm
import gym
from gym import spaces
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns


# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities for plotting and metrics tracking
from my_baselines.utils import plot_learning_curve
from my_baselines.baseline_metrics import BaselineMetricsTracker
from my_baselines.q_learning import QLearningAgent
from my_baselines.ddqn_count import DDQNCountAgent
from my_baselines.rnd import RNDAgent

# Import environment
from maze_experiment.complex_maze_env import ComplexMazeEnv

# Use importlib for importing from directory with hyphen
import importlib.util
import os

# Dynamic import for the key-maze module (since it has a hyphen in the name)
keymaze_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'key-maze', 'key_door_maze_env.py')
spec = importlib.util.spec_from_file_location('key_door_maze_env', keymaze_path)
key_door_maze_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(key_door_maze_module)
KeyDoorMazeEnv = key_door_maze_module.KeyDoorMazeEnv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a baseline agent on maze environments")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="complex_maze", choices=["complex_maze", "key_maze"],
                        help="Environment to train on")
    parser.add_argument("--maze_size", type=int, default=10,
                        help="Size of the maze (NxN)")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum steps per episode")
    parser.add_argument("--fixed_layout", action="store_true",
                        help="Use fixed maze layout")
    
    # Agent settings
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ddqn", "ddqn_count", "rnd", "tabular_q"],
                        help="Agent to train")
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden layer size for neural networks')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Initial epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.1,
                        help='Final epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=int, default=10000,
                        help='Number of steps for epsilon decay')
    parser.add_argument('--target_update', type=int, default=1000,
                        help='Update target network every n steps')
    parser.add_argument('--memory_size', type=int, default=50000,
                        help='Size of replay buffer')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for learning')
    
    # Training parameters (matching DCEO)
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Number of steps per iteration')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--render', action='store_true',
                        help='Render training')
    parser.add_argument('--render_freq', type=int, default=1,
                        help='Render every n episodes')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Evaluate every n iterations')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                        help='Save checkpoint every n iterations')
    # This parameter is already defined above
    # parser.add_argument('--max_steps', type=int, default=200,
    #                    help='Maximum number of steps per episode')
    parser.add_argument('--output_dir', type=str, default='baseline_results',
                        help='Base directory to save results')
    
    # Count-based exploration parameters (for DDQN-Count)
    parser.add_argument('--count_beta', type=float, default=0.1,
                        help='Beta parameter for count-based exploration')
    parser.add_argument('--intrinsic_weight', type=float, default=0.5,
                        help='Weight for intrinsic rewards')
                        
    # Add parameters for training that were referenced but not defined
    parser.add_argument('--buffer_size', type=int, default=50000,
                        help='Size of replay buffer')
    parser.add_argument('--update_freq', type=int, default=4,
                        help='Update frequency for network')
    parser.add_argument('--target_update_freq', type=int, default=1000,
                        help='Target network update frequency')
    
    # Optional visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Enable additional visualizations (like DCEO)')
    
    return parser.parse_args()

def create_key_maze_env(maze_size, fixed_layout=False, seed=None):
    """Create a key maze environment."""
    print(f"Creating key_maze environment with maze_size={maze_size}")
    try:
        # Corrected import path - using 'key-maze' with hyphen (directory name) instead of 'key_maze' with underscore
        from key_maze.key_door_maze_env import KeyDoorMazeEnv
    except ImportError:
        # Try the alternative import path with hyphen
        try:
            import sys
            import os
            # Add the key-maze directory to the Python path
            sys.path.append(os.path.join(os.getcwd(), 'key-maze'))
            from key_door_maze_env import KeyDoorMazeEnv
            print("Successfully imported KeyDoorMazeEnv from key-maze directory")
        except Exception as e:
            print(f"Error importing KeyDoorMazeEnv: {e}")
            return None
            
    try:
        # We'll set num_keys=1 for simplicity in initial tests
        env = KeyDoorMazeEnv(
            maze_size=maze_size, 
            num_keys=1, 
            max_steps=200,
            use_fixed_seed=True if seed is not None else False,
            fixed_seed=seed,
            use_fixed_layout=fixed_layout
        )
        print("Environment created successfully")
        # Fix for KeyDoorMazeEnv - reset once to get the proper state shape
        initial_state = env.reset()
        if initial_state is not None:
            print(f"Initial state shape: {np.array(initial_state).shape}")
        return env
    except Exception as e:
        print(f"Error creating key_maze environment: {e}")
        return None
def create_env(args):
    """Create and initialize environment."""
    print(f"\nCreating {args.env} environment with maze_size={args.maze_size}")
    
    # Create environment based on the environment type
    if args.env == 'complex_maze':
        # Import ComplexMazeEnv from the correct path
        from maze_experiment.complex_maze_env import ComplexMazeEnv
        env = ComplexMazeEnv(
            maze_size=args.maze_size,
            max_steps=args.max_steps,
            use_fixed_layout=args.fixed_layout,
            use_fixed_seed=True if args.seed is not None else False,
            fixed_seed=args.seed if args.seed is not None else 42
        )
    elif args.env == 'key_maze':
        # Direct import for the KeyDoorMazeEnv
        import sys
        import os
        # Ensure the key-maze directory is in the path
        key_maze_path = os.path.join(os.getcwd(), 'key-maze')
        if key_maze_path not in sys.path:
            sys.path.append(key_maze_path)
            
        try:
            from key_door_maze_env import KeyDoorMazeEnv
            from my_baselines.key_maze_wrapper import KeyMazeWrapper
            print("Successfully imported KeyDoorMazeEnv")
            
            # Create the base environment
            base_env = KeyDoorMazeEnv(
                maze_size=args.maze_size,
                num_keys=1,
                max_steps=200,
                use_fixed_seed=args.seed is not None,
                fixed_seed=args.seed,
                use_fixed_layout=args.fixed_layout
            )
            
            # Wrap the environment to normalize state representations
            env = KeyMazeWrapper(base_env)
            print("KeyDoorMazeEnv created and wrapped successfully")
        except Exception as e:
            print(f"Error creating KeyDoorMazeEnv: {e}")
            raise
    else:
        raise ValueError(f"Unknown environment type: {args.env}")
        
    # Set random seed for reproducibility if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if hasattr(env, 'seed'):
            env.seed(args.seed)
        
    return env

def create_agent(args, env):
    """Create and initialize agent."""
    # Get observation shape
    if hasattr(env, 'observation_space'):
        if isinstance(env.observation_space, spaces.Box):
            original_shape = env.observation_space.shape
            
            # Check if the shape is in (height, width, channels) format and convert to (channels, height, width)
            if len(original_shape) == 3 and original_shape[2] <= 5:  # Assume last dimension is channels if it's small
                input_shape = (original_shape[2], original_shape[0], original_shape[1])
                print(f"Converting shape from {original_shape} to {input_shape} for {args.env}")
            else:
                input_shape = original_shape
                print(f"Using input shape: {input_shape} for {args.env}")
        else:
            # Default shape for image-based environments
            input_shape = (3, args.maze_size, args.maze_size)
    else:
        # Default shape for non-standard environments
        input_shape = (3, args.maze_size, args.maze_size)
    
    # Get action space size
    if hasattr(env, 'action_space'):
        num_actions = env.action_space.n
    else:
        # Default to 4 actions (up, down, left, right)
        num_actions = 4
    
    # Create agent based on algorithm choice
    if args.agent == 'tabular_q' and args.env == 'key_maze':
        # Use tabular Q-learning for key maze environment
        from my_baselines.tabular_q import TabularQAgent
        print("Creating TabularQAgent for key maze environment")
        agent = TabularQAgent(
            num_actions=num_actions,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            count_bonus_weight=args.intrinsic_weight
        )
    elif args.agent == 'dqn':
        from my_baselines.dqn import DQNAgent
        agent = DQNAgent(
            input_shape=input_shape,
            num_actions=num_actions,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            update_freq=args.update_freq,
            target_update_freq=args.target_update_freq
        )
    elif args.agent == 'ddqn_count':
        from my_baselines.ddqn_count import DDQNCountAgent
        agent = DDQNCountAgent(
            input_shape=input_shape,
            num_actions=num_actions,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            update_every=args.update_freq,
            target_update=args.target_update_freq,
            count_beta=args.count_beta,
            intrinsic_weight=args.intrinsic_weight
        )
    elif args.agent == 'rnd':
        from my_baselines.rnd import RNDAgent
        agent = RNDAgent(
            input_shape=input_shape,
            num_actions=num_actions,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            update_every=args.update_freq,  # RND uses update_every instead of update_freq
            target_update=args.target_update_freq,  # RND uses target_update instead of target_update_freq
            intrinsic_weight=args.intrinsic_weight
        )
    elif args.agent == 'tabular_q':
        from my_baselines.tabular_q import TabularQAgent
        agent = TabularQAgent(
            num_actions=num_actions,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            count_bonus_weight=args.count_beta if hasattr(args, 'count_beta') else 0.1
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    
    return agent

def train(args, env, agent):
    """Train the agent on the environment."""
    # Create output directory in baseline_results folder with clear model and environment info
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = "baseline_results"
    model_env_dir = f"{args.agent}_{args.env}_maze{args.maze_size}{'_fixed' if args.fixed_layout else ''}"
    output_dir = os.path.join(base_dir, model_env_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nResults will be saved to: {output_dir}")
    
    # Create dedicated checkpoint directories inside the output directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize metrics tracker with same metrics as DCEO implementation
    # Set attributes on agent and env for consistent naming
    agent.name = args.agent
    env.name = args.env
    
    # Create metrics tracker using the exact same output_dir path
    metrics_tracker = BaselineMetricsTracker(
        env=env,
        agent=agent,
        config=vars(args),
        base_dir=""  # Empty base_dir since we'll override the results_dir anyway
    )
    
    # Override results directory to match our model directory exactly
    metrics_tracker.results_dir = output_dir
    
    # Recreate subdirectories under this path
    metrics_tracker.dirs = {
        'performance': os.path.join(output_dir, "performance_metrics"),
        'exploration': os.path.join(output_dir, "exploration_metrics"),
        'plots': os.path.join(output_dir, "plots")
    }
    
    # Create subdirectories
    for dir_path in list(metrics_tracker.dirs.values()):
        os.makedirs(dir_path, exist_ok=True)
        
    # Print confirmation to ensure it's using the right path
    print(f"\nMetrics and plots will be saved to: {output_dir}")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    solved_episodes = []
    steps_to_solve = []
    total_steps = 0
    total_episodes = 0
    
    # Make sure matplotlib is imported for rendering
    if args.render:
        import matplotlib.pyplot as plt
    
    # Training loop - Match DCEO structure with iterations and steps
    for iteration in tqdm(range(args.iterations), desc="Training"):
        # Reset environment at the start of each iteration
        state = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False
        
        # Track iteration stats
        iteration_start_step = total_steps
        iteration_episodes = 0
        iteration_rewards = []
        iteration_solved = 0
        solved = False
        
        # Run for fixed number of steps per iteration
        while total_steps - iteration_start_step < args.steps:
            # Render if enabled (matching DCEO implementation)
            if args.render and episode_step % 5 == 0:  # Only render every 5 steps to improve performance
                env.render()
                # Small pause to allow visualization to render properly
                plt.pause(0.01)
            
            # Select and perform action
            action = agent.select_action(state)
            
            # Select action without debug prints
            # Execute action in environment
            if hasattr(env, 'step'):
                try:
                    step_result = env.step(action)
                    
                    # Handle different returns from step function (old vs new gym API)
                    if len(step_result) == 5:  # New Gym API: next_state, reward, done, truncated, info
                        next_state, reward, done, truncated, info = step_result
                        done = done or truncated  # Treat truncated as done for RL training
                    else:  # Old Gym API: next_state, reward, done, info
                        next_state, reward, done, info = step_result
                        
                except Exception as e:
                    print(f"Error during environment step: {e}")
                    done = True
                    next_state = state
                    reward = 0
                    info = {}
            
            # Update agent
            agent.step(state, action, reward, next_state, done, total_episodes)
            
            # Update statistics
            episode_reward += reward
            episode_step += 1
            total_steps += 1
            
            # Update metrics tracker with current episode reward
            metrics_tracker.update_metrics(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info={
                    'episode_length': episode_step, 
                    'success': done and reward > 0,
                    'episode_reward': episode_reward  # Pass current accumulated episode reward
                }
            )
            
            state = next_state
            
            # Check if episode is solved (goal reached)
            solved = False
            if done and reward > 0:  # Assuming positive reward means reaching the goal
                solved = True
                iteration_solved += 1
                if total_episodes not in solved_episodes:
                    solved_episodes.append(total_episodes)
                    steps_to_solve.append(episode_step)
            
            if done:
                # Reset the environment for a new episode within the iteration
                state = env.reset()
                total_episodes += 1
                iteration_episodes += 1
                iteration_rewards.append(episode_reward)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                
                # Reset episode stats
                episode_reward = 0
                episode_step = 0
                done = False
        
        # After finishing an iteration, print summary and save checkpoint
        avg_reward = np.mean(iteration_rewards) if iteration_rewards else 0
        print(f"\nIteration {iteration+1}/{args.iterations} | "
              f"Steps: {total_steps} | "
              f"Episodes: {iteration_episodes} | "
              f"Solved: {iteration_solved}/{iteration_episodes} | "
              f"Avg Reward: {avg_reward:.2f}")
        
        # Save checkpoint every n iterations
        if iteration % args.checkpoint_freq == 0:
            checkpoint_name = f"{args.agent}_{args.env}_maze{args.maze_size}{'_fixed' if args.fixed_layout else ''}_iter{iteration}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            # Use agent's save method if available
            if hasattr(agent, 'save'):
                agent.save(checkpoint_path)
            else:
                # Generic saving method for all agent types
                checkpoint_data = {
                    'total_steps': total_steps,
                    'total_episodes': total_episodes,
                    'iteration': iteration,
                    'episode_rewards': episode_rewards,
                    'solved_episodes': solved_episodes,
                    'steps_to_solve': steps_to_solve,
                }
                
                # Add agent-specific data
                if args.agent == 'dqn' or args.agent == 'ddqn_count':
                    checkpoint_data.update({
                        'q_network_state_dict': agent.q_network.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'epsilon': agent.epsilon
                    })
                    
                    # Add target network for agents that use it
                    if hasattr(agent, 'target_network'):
                        checkpoint_data['target_network_state_dict'] = agent.target_network.state_dict()
                        
                elif args.agent == 'rnd':
                    checkpoint_data.update({
                        'rnd_predictor_state_dict': agent.rnd_predictor.state_dict(),
                        'q_network_state_dict': agent.q_network.state_dict(),
                        'target_network_state_dict': agent.target_network.state_dict(),
                        'q_optimizer_state_dict': agent.q_optimizer.state_dict(),
                        'rnd_optimizer_state_dict': agent.rnd_optimizer.state_dict(),
                        'epsilon': agent.epsilon
                    })
                    
                # Save the checkpoint
                torch.save(checkpoint_data, checkpoint_path)
            
            print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Save checkpoint at the end of training
    model_name = f"{args.agent}_{args.env}_maze{args.maze_size}{'_fixed' if args.fixed_layout else ''}_final.pt"
    model_save_path = os.path.join(output_dir, model_name)
    
    # For all agent types
    if hasattr(agent, 'save'):
        agent.save(model_save_path)
    else:
        # Generic saving method for all agent types
        checkpoint_data = {
            'total_steps': total_steps,
            'total_episodes': total_episodes,
            'iteration': args.iterations,  # Mark as final iteration
            'episode_rewards': episode_rewards,
            'solved_episodes': solved_episodes,
            'steps_to_solve': steps_to_solve,
        }
        
        # Add agent-specific data
        if args.agent == 'dqn' or args.agent == 'ddqn_count':
            checkpoint_data.update({
                'q_network_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon
            })
            
            # Add target network for agents that use it
            if hasattr(agent, 'target_network'):
                checkpoint_data['target_network_state_dict'] = agent.target_network.state_dict()
                
        elif args.agent == 'rnd':
            checkpoint_data.update({
                'q_network_state_dict': agent.q_network.state_dict() if hasattr(agent, 'q_network') else None,
                'predictor_network_state_dict': agent.predictor_network.state_dict() if hasattr(agent, 'predictor_network') else None,
                'target_network_state_dict': agent.target_network.state_dict() if hasattr(agent, 'target_network') else None,
                'rnd_predictor_state_dict': agent.rnd_predictor.state_dict() if hasattr(agent, 'rnd_predictor') else None,
                'rnd_target_state_dict': agent.rnd_target.state_dict() if hasattr(agent, 'rnd_target') else None,
                'optimizer_state_dict': agent.optimizer.state_dict() if hasattr(agent, 'optimizer') else None
            })
            
        # Save the checkpoint
        torch.save(checkpoint_data, model_save_path)
    
    print(f"\nFinal model saved to: {model_save_path}")
    
    # Generate final plots and save metrics
    metrics_tracker.plot_all_metrics()
    metrics_tracker.save_all_metrics()
    
    # Additional custom plots from the original script if needed
    try:
        plot_training_results(args, agent, output_dir, episode_rewards, solved_episodes, steps_to_solve)
    except Exception as e:
        print(f"Note: Some additional custom plots could not be generated: {e}")
    
    # Calculate total training time
    total_time = time.time() - metrics_tracker.start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed: {total_episodes} episodes over {total_steps} steps")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results saved in: {output_dir}")

def plot_training_results(args, agent, output_dir, episode_rewards, solved_episodes, steps_to_solve):
    """Plot and save training results."""
    # Create a plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot epsilon over time if available
    if hasattr(agent, 'epsilons') and len(agent.epsilons) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(agent.epsilons)
        plt.title(f'{args.agent.upper()} - Epsilon Decay')
        plt.xlabel('Step')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'epsilon_decay.png'))
        plt.close()
    
    # Plot episode rewards
    if len(episode_rewards) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title(f'{args.agent.upper()} - Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'episode_rewards.png'))
        plt.close()
        
        # Plot moving average of rewards
        plt.figure(figsize=(10, 5))
        window_size = min(20, max(5, len(episode_rewards) // 10))
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg)
        plt.title(f'{args.agent.upper()} - Moving Average of Rewards (Window: {window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'rewards_moving_avg.png'))
        plt.close()
        # Save solved episodes data
        np.save(os.path.join(output_dir, "solved_episodes.npy"), np.array(solved_episodes))
        np.save(os.path.join(output_dir, "steps_to_solve.npy"), np.array(steps_to_solve))
    
    # Save all rewards and losses
    np.save(os.path.join(output_dir, "episode_rewards.npy"), np.array(agent.episode_rewards))
    if hasattr(agent, 'losses'):
        np.save(os.path.join(output_dir, "losses.npy"), np.array(agent.losses))
    if hasattr(agent, 'intrinsic_rewards'):
        np.save(os.path.join(output_dir, "intrinsic_rewards.npy"), np.array(agent.intrinsic_rewards))

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Print training configuration
    print("\n===== BASELINE TRAINING CONFIGURATION =====")
    print(f"Algorithm: {args.agent.upper()}")
    print(f"Environment: {args.env}")
    print(f"Maze size: {args.maze_size}x{args.maze_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Steps per iteration: {args.steps}")
    print(f"Fixed layout: {args.fixed_layout}")
    print(f"Random seed: {args.seed}")
    print(f"Rendering: {args.render}")
    print("=========================================\n")
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")
    
    # Create environment
    env = create_env(args)
    
    # Create agent
    agent = create_agent(args, env)
    
    # Record start time
    start_time = time.time()
    
    # Train agent
    train(args, env, agent)
    
    # Record end time and print training duration
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
