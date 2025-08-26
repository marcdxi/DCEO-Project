"""
Training script for the fully online Rainbow DCEO agent on the Key-Door Maze environment.
This version includes enhanced visualization similar to compare_key_maze_agents.py.
"""

import numpy as np
import torch
import time
import os
import sys
import argparse
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import clear_output
import seaborn as sns

# Import metrics tracker
from key_maze_metrics_new import KeyMazeMetricsTracker

# Import PyTorch online DCEO implementation
from pytorch_dceo_online import FullyOnlineDCEOAgent

# Import Key-Door Maze environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'key-maze'))
from key_door_maze_env import KeyDoorMazeEnv

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess_state(state, shape=(84, 84)):
    """Preprocess the state from the environment for the agent."""
    import numpy as np
    import cv2
    
    # Handle gym's new API which returns (obs, info) tuples
    if isinstance(state, tuple):
        state = state[0]  # Extract observation
    
    # For Key-Door Maze environment, state is a multi-channel grid
    if not isinstance(state, np.ndarray):
        try:
            state = np.array(state, dtype=np.float32)
        except ValueError:
            print(f"Error converting state to numpy array. State type: {type(state)}")
            if isinstance(state, tuple):
                print(f"State tuple length: {len(state)}")
                for i, item in enumerate(state):
                    print(f"Item {i} type: {type(item)}")
            raise
    
    # Create an RGB image from the multi-channel state
    # Channel 0: Agent location (red)
    # Channel 1: Walls (black)
    # Channel 2: Goal (green)
    # Channel 3: Keys (yellow)
    # Channel 4: Doors (brown)
    
    # Create a blank RGB image
    h, w = state.shape[:2]  # Height and width of the maze
    rgb_image = np.ones((h, w, 3), dtype=np.float32)  # White background
    
    # Add colors for each channel
    rgb_image[state[:,:,1] > 0] = [0.0, 0.0, 0.0]  # Walls (black)
    rgb_image[state[:,:,2] > 0] = [0.0, 1.0, 0.0]  # Goal (green)
    rgb_image[state[:,:,3] > 0] = [1.0, 1.0, 0.0]  # Keys (yellow)
    rgb_image[state[:,:,4] > 0] = [0.7, 0.3, 0.0]  # Doors (brown)
    rgb_image[state[:,:,0] > 0] = [1.0, 0.0, 0.0]  # Agent (red)
    
    # Resize to target shape
    if (h, w) != shape:
        rgb_image = cv2.resize(rgb_image, shape, interpolation=cv2.INTER_AREA)
    
    # Convert from (H, W, C) to (C, H, W) format for PyTorch
    rgb_image = rgb_image.transpose(2, 0, 1).astype(np.float32)
    
    return rgb_image


def handle_env_step(env, action):
    """Universal step function that works with both old and new Gym APIs."""
    step_result = env.step(action)
    
    # Handle different step() return formats
    if isinstance(step_result, tuple):
        if len(step_result) == 5:  # New Gymnasium API: (next_state, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            return next_state, reward, done, info
        else:  # Old Gym API: (next_state, reward, done, info)
            return step_result
    else:
        raise ValueError(f"Unexpected step result type: {type(step_result)}")


def handle_env_reset(env, **kwargs):
    """Universal reset function that works with both old and new Gym APIs."""
    reset_result = env.reset(**kwargs)
    
    # Handle different reset() return formats
    if isinstance(reset_result, tuple):
        if len(reset_result) == 2:  # New Gymnasium API: (state, info)
            state, _ = reset_result
            return state
        else:
            return reset_result[0]  # Just return the first element
    else:  # Old Gym API: state
        return reset_result


def evaluate_agent(agent, env, num_episodes=5, fig=None, axes=None, plot_metrics=None):
    """Evaluate the agent and visualize performance."""
    total_rewards = []
    total_steps = []
    success_count = 0
    all_keys_collected = []
    all_doors_opened = []
    
    # Create figure and axes if not provided
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        main_ax, metrics_ax = axes
    else:
        main_ax, metrics_ax = axes
    
    # Clear the axes
    main_ax.clear()
    metrics_ax.clear()
    
    # Set up metrics plot
    if plot_metrics is None:
        plot_metrics = {'rewards': [], 'success_rates': [], 'iterations': []}
    
    metrics_ax.set_title('Training Metrics')
    metrics_ax.set_xlabel('Iteration')
    metrics_ax.set_ylabel('Value')
    
    # Plot metrics
    if len(plot_metrics['iterations']) > 0:
        metrics_ax.plot(plot_metrics['iterations'], plot_metrics['rewards'], 'b-', label='Average Reward')
        metrics_ax.plot(plot_metrics['iterations'], plot_metrics['success_rates'], 'g-', label='Success Rate')
        metrics_ax.legend()
    
    # Real-time visualization
    plt.ion()  # Enable interactive mode
    plt.show()
    
    for episode in range(num_episodes):
        state = handle_env_reset(env)
        state = preprocess_state(state)
        
        done = False
        episode_reward = 0
        steps = 0
        keys_collected = 0
        doors_opened = 0
        
        while not done:
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.select_action(state_tensor, eval_mode=True)
            
            # Take action in environment
            next_state, reward, done, info = handle_env_step(env, action)
            next_state = preprocess_state(next_state)
            
            # Update tracking variables
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Track keys and doors
            if info.get('key_collected', False):
                keys_collected += 1
                print(f"Episode {episode+1}: Got key {keys_collected} at step {steps}")
            if info.get('door_opened', False):
                doors_opened += 1
                print(f"Episode {episode+1}: Opened door {doors_opened} at step {steps}")
            if info.get('goal_reached', False):
                print(f"Episode {episode+1}: Reached goal at step {steps}! Success!")
            
            # Get Maze image
            maze_img = env.render(mode='rgb_array')
            if maze_img is not None:
                # Draw maze
                main_ax.clear()
                main_ax.imshow(maze_img)
                
                # Add title with information
                title = f"Evaluation - Episode {episode+1}/{num_episodes}, Step {steps}\n"
                title += f"Reward: {episode_reward:.2f}, Keys: {keys_collected}/{env.unwrapped.num_keys}, "
                title += f"Doors: {doors_opened}/{len(env.unwrapped.door_positions) if hasattr(env.unwrapped, 'door_positions') else '?'}"
                main_ax.set_title(title)
                
                # Update plot
                plt.draw()
                plt.pause(0.01)
            
            # Log success
            if info.get('goal_reached', False):
                success_count += 1
                print(f"Episode {episode+1}: Success!")
                break
        
        # Record episode statistics
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        all_keys_collected.append(keys_collected)
        all_doors_opened.append(doors_opened)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    # Print summary statistics
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = success_count / num_episodes
    
    print(f"\nEvaluation summary:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Success rate: {success_rate:.2f} ({success_count}/{num_episodes})")
    
    return avg_reward, success_rate, fig, axes, plot_metrics


def train_agent(config, num_iterations=100, eval_frequency=10, num_train_steps=10000, render_training=False):
    """Train the fully online Rainbow DCEO agent with enhanced visualization."""
    
    # Create environment
    env = KeyDoorMazeEnv(
        maze_size=config['maze_size'],
        num_keys=config['num_keys'],
        max_steps=200,  # Fixed at 200 steps as per DCEO paper
        use_fixed_seed=config.get('use_fixed_seed', False),
        fixed_seed=config.get('fixed_seed', None),
        use_fixed_layout=config.get('use_fixed_layout', False)
    )
    
    # Initialize agent with fully online DCEO implementation
    agent = FullyOnlineDCEOAgent(
        input_shape=config['input_shape'],
        num_actions=config['num_actions'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        update_horizon=config['update_horizon'],
        min_replay_history=config['min_replay_history'],
        update_period=config['update_period'],
        target_update_period=config['target_update_period'],
        epsilon_train=config['epsilon_train'],
        epsilon_eval=config['epsilon_eval'],
        epsilon_decay_period=config['epsilon_decay_period'],
        learning_rate=config['learning_rate'],
        noisy=config['noisy'],
        dueling=config['dueling'],
        double_dqn=config['double_dqn'],
        distributional=config['distributional'],
        num_atoms=config['num_atoms'],
        v_min=config['v_min'],
        v_max=config['v_max'],
        num_options=config['num_options'],
        option_prob=config['option_prob'],
        option_duration=config['option_duration'],
        rep_dim=config['rep_dim'],
        log_transform=config['log_transform'],
        orthonormal=config.get('orthonormal', True),
        alpha_rep=config.get('alpha_rep', 1.0),
        alpha_main=config.get('alpha_main', 1.0),
        alpha_option=config.get('alpha_option', 1.0)
    )
    
    # Initialize metrics tracker
    metrics_tracker = KeyMazeMetricsTracker(env, agent, config)
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # For tracking progress
    best_success_rate = 0.0
    total_training_steps = 0
    training_rewards = []
    eval_rewards = []
    success_rates = []
    iterations = []
    episodes_completed = 0
    
    # Set up visualization only if rendering is enabled
    fig = None
    axes = None
    main_ax = None 
    metrics_ax = None
    
    if render_training:
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        main_ax, metrics_ax = axes
    
    # Metrics for plotting
    plot_metrics = {
        'rewards': [],
        'success_rates': [],
        'iterations': []
    }
    
    # Start time
    start_time = time.time()
    
    for iteration in range(1, num_iterations + 1):
        # Reset environment for next episode
        state = handle_env_reset(env)
        state = preprocess_state(state)
        done = False
        episode_reward = 0
        episode_steps = 0
        keys_collected = 0
        doors_opened = 0
        episodes_completed += 1
        
        # Show reset state
        if render_training:
            maze_img = env.render(mode='rgb_array')
            if maze_img is not None:
                main_ax.clear()
                main_ax.imshow(maze_img)
                epsilon = agent._get_epsilon()
                title = f"Training - Iteration {iteration}/{num_iterations}, New Episode {episodes_completed} (Total Steps: {total_training_steps})\n"
                title += f"Keys: 0/{env.unwrapped.num_keys}, Doors: 0/{len(env.unwrapped.door_positions)}, Epsilon: {epsilon:.4f}"
                main_ax.set_title(title)
                plt.draw()
                plt.pause(0.01)
        
        option_changes = 0
        last_option = None
        
        print(f"\nIteration {iteration}/{num_iterations} - Training")
        
        # Initialize steps counter for this iteration
        steps_this_iteration = 0
        
        for step in range(1, num_train_steps + 1):
            # Get current option
            current_option = agent.cur_opt
            if last_option is not None and current_option != last_option:
                option_changes += 1
            last_option = current_option
            
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.select_action(state_tensor)
            
            # Take action in environment
            next_state, reward, done, info = handle_env_step(env, action)
            next_state = preprocess_state(next_state)
            
            # Track keys and doors
            if info.get('key_collected', False):
                keys_collected += 1
                print(f"  Step {episode_steps}: Collected key {keys_collected}")
            if info.get('door_opened', False):
                doors_opened += 1
                print(f"  Step {episode_steps}: Opened door {doors_opened}")
            if info.get('goal_reached', False):
                print(f"  Step {episode_steps}: Reached goal! Success!")
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent - fully online update
            agent.update()
            
            # Update tracking variables
            episode_reward += reward
            episode_steps += 1
            total_training_steps += 1
            
            # Track metrics for this step
            metrics_tracker.log_training_step(state, action, reward, next_state, done, info, total_training_steps, episode_steps)
            
            state = next_state
            
            # Visualize every few steps or on important events (only if rendering is enabled)
            if render_training and main_ax is not None and (step % 5 == 0 or info.get('key_collected', False) or 
                                  info.get('door_opened', False) or done):
                # Get maze image
                maze_img = env.render(mode='rgb_array')
                if maze_img is not None:
                    # Draw maze
                    main_ax.clear()
                    main_ax.imshow(maze_img)
                    
                    # Add title with information
                    epsilon = agent._get_epsilon()
                    title = f"Training - Iteration {iteration}/{num_iterations}, Step {episode_steps} (Total: {total_training_steps})\n"
                    title += f"Episode: {episodes_completed}, Reward: {episode_reward:.2f}\n"
                    title += f"Keys: {keys_collected}/{env.unwrapped.num_keys}, Doors: {doors_opened}/{len(env.unwrapped.door_positions)}, Epsilon: {epsilon:.4f}\n"
                    
                    if current_option is not None:
                        title += f"Current Option: {current_option}, Option Changes: {option_changes}\n"
                    main_ax.set_title(title)
                    
                    # Show current status in the metrics plot
                    if metrics_ax is not None:
                        metrics_ax.clear()
                        if plot_metrics['iterations']:
                            metrics_ax.plot(plot_metrics['iterations'], plot_metrics['rewards'], 'b-', label='Avg Reward')
                            metrics_ax.plot(plot_metrics['iterations'], plot_metrics['success_rates'], 'g-', label='Success Rate')
                            metrics_ax.set_title('Training Progress')
                            metrics_ax.set_xlabel('Iteration')
                            metrics_ax.set_ylabel('Value')
                            metrics_ax.legend()
                            metrics_ax.grid(True)
                        else:
                            metrics_ax.text(0.5, 0.5, "No metrics yet", horizontalalignment='center', verticalalignment='center')
                        
                        # Update display
                        plt.draw()
                        plt.pause(0.001)
            
            # Episode completed or truncated
            if done:
                training_rewards.append(episode_reward)
                
                # Print progress
                if episodes_completed % 10 == 0:
                    print(f"Episode {episodes_completed} - Reward: {episode_reward:.2f}, Steps: {episode_steps}")
                
                # Log metrics for this episode
                metrics_tracker.log_episode_end(episode_reward, episode_steps, iteration)
                
                # Reset environment for next episode
                state = handle_env_reset(env)
                state = preprocess_state(state)
                done = False
                episode_reward = 0
                episode_steps = 0
                keys_collected = 0
                doors_opened = 0
                episodes_completed += 1
            
            # Save model and display progress periodically
            if iteration % eval_frequency == 0:
                # Calculate current success rate from recent episodes
                window_size = 5  # Same as used in metrics tracking
                if len(metrics_tracker.performance_metrics['episode_success']) >= window_size:
                    current_success_rate = sum(metrics_tracker.performance_metrics['episode_success'][-window_size:]) / window_size
                    
                    # Update tracking metrics for visualization
                    eval_rewards.append(np.mean(metrics_tracker.performance_metrics['episode_rewards'][-window_size:]))
                    success_rates.append(current_success_rate)
                    iterations.append(iteration)
                    
                    # Update plot metrics
                    plot_metrics['rewards'] = eval_rewards
                    plot_metrics['success_rates'] = success_rates
                    plot_metrics['iterations'] = iterations
                    
                    # Save best model if better than previous best
                    if current_success_rate > best_success_rate:
                        best_success_rate = current_success_rate
                        best_model_path = os.path.join(config['checkpoint_dir'], "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        agent.save(best_model_path)
                        print(f"\nSaved best model (success rate: {current_success_rate:.2f}) at {best_model_path}")
                    
                    # Display current progress
                    current_step = metrics_tracker.performance_metrics['total_env_steps']
                    print(f"\nIteration {iteration}/{num_iterations} - Status Update")
                    print(f"Environment steps: {current_step}")
                    print(f"Current success rate: {current_success_rate:.2f}")
                    print(f"State coverage: {len(metrics_tracker.exploration_metrics['unique_states_visited'])} unique states")
                    
                    # Save metrics and generate intermediate plots
                    metrics_tracker.save_metrics()
                    metrics_tracker._generate_milestone_plots(current_step)
                print(f"Environment steps: {current_step}")
                print(f"Current success rate: {current_success_rate:.2f}")
                print(f"State coverage: {len(metrics_tracker.exploration_metrics['unique_states_visited'])} unique states")
                
                # Save metrics and generate plots periodically
                metrics_tracker.save_metrics()
                metrics_tracker._generate_milestone_plots(current_step)

    # Print time elapsed
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")

    # Final update - save metrics and generate final plots
    print("\nTraining Complete")
    
    # Save final metrics and generate comprehensive plots
    metrics_tracker.save_metrics()
    metrics_tracker.generate_plots()
    
    # Save final model
    final_model_path = os.path.join(config['checkpoint_dir'], "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    agent.save(final_model_path)
    print(f"Final model saved at {final_model_path}")
    print(f"Final metrics and plots saved in {metrics_tracker.results_dir}")
    
    # Print final statistics aligned with DCEO paper metrics
    print("\nFinal Statistics:")
    total_states = env.unwrapped.maze_size * env.unwrapped.maze_size
    coverage = len(metrics_tracker.exploration_metrics['unique_states_visited']) / total_states
    print(f"State Coverage: {coverage*100:.2f}% ({len(metrics_tracker.exploration_metrics['unique_states_visited'])}/{total_states} states)")
    
    # Option usage statistics
    print("\nOption Usage:")
    for opt_idx in range(agent.num_options):
        count = metrics_tracker.option_metrics['option_selection_counts'].get(opt_idx, 0)
        print(f"Option {opt_idx}: Selected {count} times")
    primitive_count = metrics_tracker.option_metrics['primitive_action_count']
    print(f"Primitive actions: Used {primitive_count} times")
    
    # Key performance metrics
    print("\nPerformance:")
    if metrics_tracker.performance_metrics['episode_success']:
        recent_success = sum(metrics_tracker.performance_metrics['episode_success'][-10:]) / min(10, len(metrics_tracker.performance_metrics['episode_success']))
        print(f"Final success rate: {recent_success*100:.2f}%")
    if metrics_tracker.performance_metrics['episode_rewards']:
        recent_reward = sum(metrics_tracker.performance_metrics['episode_rewards'][-10:]) / min(10, len(metrics_tracker.performance_metrics['episode_rewards']))
        print(f"Final average reward: {recent_reward:.2f}")
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Total training steps: {total_training_steps}")
    print(f"Best success rate: {best_success_rate:.2f}")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    
    # Plot final metrics
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, eval_rewards, 'b-', label='Average Reward')
    ax.plot(iterations, success_rates, 'g-', label='Success Rate')
    ax.set_title('Training Metrics')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    plt.savefig(os.path.join(config['checkpoint_dir'], 'training_metrics.png'))
    plt.close()
    
    # Close environment
    env.close()
    
    return agent
    
    # Print final statistics aligned with DCEO paper metrics
    print("\nFinal Statistics:")
    total_states = env.unwrapped.maze_size * env.unwrapped.maze_size
    coverage = len(metrics_tracker.exploration_metrics['unique_states_visited']) / total_states
    print(f"State Coverage: {coverage*100:.2f}% ({len(metrics_tracker.exploration_metrics['unique_states_visited'])}/{total_states} states)")
    
    # Option usage statistics
    print("\nOption Usage:")
    for opt_idx in range(agent.num_options):
        count = metrics_tracker.option_metrics['option_selection_counts'].get(opt_idx, 0)
        print(f"Option {opt_idx}: Selected {count} times")
    primitive_count = metrics_tracker.option_metrics['primitive_action_count']
    print(f"Primitive actions: Used {primitive_count} times")
    
    # Key performance metrics
    print("\nPerformance:")
    if metrics_tracker.performance_metrics['episode_success']:
        recent_success = sum(metrics_tracker.performance_metrics['episode_success'][-10:]) / min(10, len(metrics_tracker.performance_metrics['episode_success']))
        print(f"Final success rate: {recent_success*100:.2f}%")
    if metrics_tracker.performance_metrics['episode_rewards']:
        recent_reward = sum(metrics_tracker.performance_metrics['episode_rewards'][-10:]) / min(10, len(metrics_tracker.performance_metrics['episode_rewards']))
        print(f"Final average reward: {recent_reward:.2f}")
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Total training steps: {total_training_steps}")
    print(f"Best success rate: {best_success_rate:.2f}")
    try:
        print(f"Final success rate: {success_rate:.2f}")
        print(f"Final average reward: {avg_reward:.2f}")
    except UnboundLocalError:
        print("Warning: Final evaluation metrics not available")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    
    # Plot final metrics
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, eval_rewards, 'b-', label='Average Reward')
    ax.plot(iterations, success_rates, 'g-', label='Success Rate')
    ax.set_title('Training Metrics')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    plt.savefig(os.path.join(config['checkpoint_dir'], 'training_metrics.png'))
    plt.close()
    
    # Close environment


def get_config(maze_size=12, num_keys=1, num_options=5, seed=None, fixed_layout=False):
    """Get configuration for the Key-Door Maze environment."""
    # Create configuration
    config = {
        # Environment params
        'maze_size': maze_size,
        'num_keys': num_keys,
        'num_actions': 4,  # Up, Right, Down, Left
        'input_shape': [3, 84, 84],  # (channels, height, width) - PyTorch format
        'use_fixed_seed': seed is not None,
        'fixed_seed': seed,
        'use_fixed_layout': fixed_layout,
        
        # Agent params
        'num_options': num_options,
        'buffer_size': 100000,
        'batch_size': 32,
        'gamma': 0.99,
        'update_horizon': 1,
        'min_replay_history': 1000,
        'update_period': 4,
        'target_update_period': 1000,
        'epsilon_train': 0.01,
        'epsilon_eval': 0.001,
        'epsilon_decay_period': 50000,
        'learning_rate': 0.0001,
        'noisy': False,
        'dueling': True,
        'double_dqn': True,
        'distributional': True,
        'num_atoms': 51,
        'v_min': -10.0,
        'v_max': 10.0,
        'option_prob': 0.5,  # Probability of selecting option vs primitive
        'option_duration': 10,  # Maximum duration of options
        'rep_dim': 64,  # Dimension of representation
        'log_transform': True,  # Whether to use log transform for representation
        'orthonormal': True,  # Whether to use orthonormal basis functions
        'alpha_rep': 1.0,  # Weight for representation loss
        'alpha_main': 1.0,  # Weight for main task loss
        'alpha_option': 1.0,  # Weight for option loss
        
        # Training params
        'checkpoint_dir': f'checkpoints/keymaze_{maze_size}x{maze_size}_{num_keys}keys_{num_options}options',
        'checkpoint_every_n_steps': 10000
    }
    
    # Update the directory name if using a fixed seed
    if seed is not None:
        config['checkpoint_dir'] += f'_seed{seed}'
    
    return config


def main():
    """Main function to run training or evaluation."""
    parser = argparse.ArgumentParser(description="Train or evaluate the fully online DCEO agent on KeyDoorMaze")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps per iteration")
    parser.add_argument("--maze_size", type=int, default=10, help="Size of the maze (minimum 8 recommended)")
    parser.add_argument("--num_keys", type=int, default=1, help="Number of keys in the maze")
    parser.add_argument("--num_options", type=int, default=5, help="Number of options for DCEO")
    parser.add_argument("--render", action="store_true", help="Render training episodes")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to load checkpoint from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--fixed_layout", action="store_true", help="Use a fixed maze layout throughout training")
    
    args = parser.parse_args()
    
    # Ensure maze size is at least 8 to avoid errors in maze generation
    if args.maze_size < 8:
        print(f"Warning: Maze size {args.maze_size} is too small and may cause errors.")
        print("Setting maze size to 8 (minimum recommended size).")
        args.maze_size = 8
    
    # Get configuration
    config = get_config(
        maze_size=args.maze_size,
        num_keys=args.num_keys,
        num_options=args.num_options,
        seed=args.seed,
        fixed_layout=args.fixed_layout
    )
    
    if args.mode == "train":
        print("Training fully online DCEO agent on KeyDoorMaze")
        agent = train_agent(
            config=config,
            num_iterations=args.iterations,
            num_train_steps=args.steps,
            render_training=args.render
        )
    
    elif args.mode == "eval":
        print("Evaluating fully online DCEO agent on KeyDoorMaze")
        
        # Create environment
        env = KeyDoorMazeEnv(
            maze_size=config['maze_size'],
            num_keys=config['num_keys'],
            max_steps=config['maze_size'] * 4
        )
        
        # Initialize agent
        agent = FullyOnlineDCEOAgent(
            input_shape=config['input_shape'],
            num_actions=config['num_actions'],
            num_options=config['num_options']
        )
        
        # Load checkpoint if provided
        if args.checkpoint:
            agent.load(args.checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print("No checkpoint provided, using randomly initialized agent")
        
        # Set up visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Evaluate agent
        evaluate_agent(agent, env, num_episodes=10, fig=fig, axes=axes)
        
        # Close environment
        env.close()


if __name__ == "__main__":
    main()
