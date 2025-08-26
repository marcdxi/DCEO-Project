"""
Training script for the fully online Rainbow DCEO agent on the Complex Maze environment.
This version includes visualization options similar to train_keymaze_online_dceo_with_viz.py.
"""

import os
import sys
import time
import json
import argparse
import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Import metrics tracker
from maze_metrics import MazeMetricsTracker

# Import PyTorch online DCEO implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorch_dceo_online import FullyOnlineDCEOAgent

# Import Complex Maze environment
from complex_maze_env import ComplexMazeEnv


def preprocess_state(state, shape=(84, 84)):
    """Preprocess the state from the environment for the agent."""
    if state is None:
        return np.zeros((3,) + shape, dtype=np.float32)
    
    # Convert to uint8 if needed for PIL operations
    if state.dtype != np.uint8 and np.issubdtype(state.dtype, np.number):
        if np.max(state) <= 1.0:
            state = (state * 255).astype(np.uint8)
        else:
            state = state.astype(np.uint8)
    
    # Handle different input shapes
    if len(state.shape) == 2:  # Grayscale
        # Convert to RGB by duplicating channels
        state = np.stack([state] * 3, axis=2)
    elif len(state.shape) == 3 and state.shape[0] == 1 and state.shape[1] == 1:  # Special case from the maze env
        # Create a full-size state with the color
        full_state = np.zeros(shape + (3,), dtype=np.uint8)
        full_state[:, :] = state[0, 0]
        state = full_state
    
    # Ensure correct shape (84x84x3)
    if state.shape[:2] != shape:
        try:
            from PIL import Image
            # Convert to PIL image for resizing
            img = Image.fromarray(state)
            img = img.resize(shape, Image.BICUBIC)
            # Convert back to numpy array
            state = np.array(img)
        except Exception as e:
            print(f"Error during image resizing: {e}")
            # Fallback: create a plain state with the right shape
            state = np.zeros(shape + (3,), dtype=np.uint8)
    
    # If state has multiple channels (RGB), convert to PyTorch format (channels first)
    if len(state.shape) == 3 and state.shape[2] == 3:
        state = np.transpose(state, (2, 0, 1))
    
    # Scale pixel values to 0-1 range
    state = state.astype(np.float32) / 255.0
    
    return state


def handle_env_step(env, action):
    """Universal step function that works with both old and new Gym APIs."""
    try:
        # Try new API (step returns additional info 'truncated')
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info
    except ValueError:
        # Fallback to old API
        return env.step(action)


def handle_env_reset(env, **kwargs):
    """Universal reset function that works with both old and new Gym APIs."""
    try:
        # Try new API (reset can return info)
        reset_result = env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            return reset_result[0]  # New API returns (state, info)
        return reset_result
    except TypeError:
        # Fallback to old API
        return env.reset()


def evaluate_agent(agent, env, num_episodes=5, fig=None, axes=None, plot_metrics=None, render=False):
    """Evaluate the agent and visualize performance if render=True."""
    # Get device from agent
    if hasattr(agent, 'device'):
        device = agent.device
    else:
        # Fallback to config or default to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up visualization if enabled
    if render and fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        main_ax, metrics_ax = axes
    elif render:
        main_ax, metrics_ax = axes
    
    # Prepare evaluation metrics
    all_rewards = []
    all_success = []
    all_steps = []
    
    for episode in range(1, num_episodes + 1):
        # Reset environment and agent
        state = handle_env_reset(env)
        state = preprocess_state(state)
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Initialize visualization of this episode
        if render and main_ax is not None:
            maze_img = env.render(mode='rgb_array')
            if maze_img is not None:
                main_ax.clear()
                main_ax.imshow(maze_img)
                main_ax.set_title(f"Evaluation - Episode {episode}/{num_episodes}, Step 0")
                plt.draw()
                plt.pause(0.01)
        
        print(f"Evaluation episode {episode}/{num_episodes}")
        
        while not done:
            # Get action from agent (eval mode)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.act(state_tensor, eval_mode=True)
            
            # Take action in environment
            next_state, reward, done, info = handle_env_step(env, action)
            next_state = preprocess_state(next_state)
            
            # Update counters
            episode_steps += 1
            episode_reward += reward
            
            # Render visualization if enabled
            if render and main_ax is not None and (episode_steps % 5 == 0 or done):
                maze_img = env.render(mode='rgb_array')
                if maze_img is not None:
                    main_ax.clear()
                    main_ax.imshow(maze_img)
                    
                    # Add title with information
                    title = f"Evaluation - Episode {episode}/{num_episodes}, Step {episode_steps}\n"
                    title += f"Current Reward: {episode_reward:.2f}"
                    
                    if hasattr(agent, 'cur_opt') and agent.cur_opt is not None:
                        title += f", Option: {agent.cur_opt}"
                    
                    main_ax.set_title(title)
                    plt.draw()
                    plt.pause(0.01)
            
            # Update state
            state = next_state
            
            # Optional slowdown for visualization
            if render:
                time.sleep(0.01)
        
        # Record episode results
        success = info.get('goal_reached', False)
        all_rewards.append(episode_reward)
        all_success.append(1.0 if success else 0.0)
        all_steps.append(episode_steps)
        
        print(f"  Reward: {episode_reward:.2f}, Success: {success}, Steps: {episode_steps}")
    
    # Calculate evaluation metrics
    avg_reward = np.mean(all_rewards)
    success_rate = np.mean(all_success)
    avg_steps = np.mean(all_steps)
    
    print(f"Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Success Rate: {success_rate:.2f}")
    print(f"  Average Steps: {avg_steps:.2f}")
    
    # Update plot metrics if provided
    if plot_metrics is not None:
        if 'eval_rewards' not in plot_metrics:
            plot_metrics['eval_rewards'] = []
            plot_metrics['eval_success_rates'] = []
            
        plot_metrics['eval_rewards'].append(avg_reward)
        plot_metrics['eval_success_rates'].append(success_rate)
        
        # Update evaluation plot if visualization is enabled
        if render and metrics_ax is not None:
            metrics_ax.clear()
            x = range(1, len(plot_metrics['eval_rewards']) + 1)
            metrics_ax.plot(x, plot_metrics['eval_rewards'], 'b-', label='Avg Reward')
            metrics_ax.plot(x, plot_metrics['eval_success_rates'], 'g-', label='Success Rate')
            metrics_ax.set_title('Evaluation Results')
            metrics_ax.set_xlabel('Evaluation')
            metrics_ax.set_ylabel('Value')
            metrics_ax.legend()
            metrics_ax.grid(True)
            plt.draw()
            plt.pause(0.1)
    
    return avg_reward, success_rate, fig, axes, plot_metrics


def train_agent(config, num_iterations=100, num_train_steps=10000, render_training=False):
    """Train the fully online Rainbow DCEO agent with visualization if enabled."""
    
    # Create environment
    env = ComplexMazeEnv(
        maze_size=config['maze_size'],
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
    metrics_tracker = MazeMetricsTracker(env, agent, config)
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # For tracking progress
    best_success_rate = 0.0
    total_training_steps = 0
    episodes_completed = 0
    training_rewards = []
    
    # Get device from agent
    if hasattr(agent, 'device'):
        device = agent.device
    else:
        # Fallback to config or default to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    # Set up visualization if enabled
    fig = None
    main_ax = None
    metrics_ax = None
    if render_training:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        main_ax, metrics_ax = axes
    
    # For plotting metrics
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
        episodes_completed += 1
        
        # Show reset state if rendering is enabled
        if render_training and main_ax is not None:
            maze_img = env.render(mode='rgb_array')
            if maze_img is not None:
                main_ax.clear()
                main_ax.imshow(maze_img)
                epsilon = agent._get_epsilon()
                title = f"Training - Iteration {iteration}/{num_iterations}, New Episode {episodes_completed}\n"
                title += f"Total Steps: {total_training_steps}, Epsilon: {epsilon:.4f}"
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
            
            # Track goal reached events
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
            
            # Log wall clock time for performance monitoring
            if total_training_steps % 1000 == 0:
                elapsed = time.time() - metrics_tracker.start_time
                metrics_tracker.performance_metrics['wall_time'].append((total_training_steps, elapsed))
            
            state = next_state
            
            # Visualize once per 10 steps or on goal reached (only if rendering is enabled)
            if render_training and main_ax is not None and (step % 10 == 0 or info.get('goal_reached', False) or done):
                # Get maze image
                maze_img = env.render(mode='rgb_array')
                if maze_img is not None:
                    # Draw maze
                    main_ax.clear()
                    main_ax.imshow(maze_img)
                    
                    # Add title with information
                    epsilon = agent._get_epsilon()
                    title = f"Training - Iteration {iteration}/{num_iterations}, Step {episode_steps}\n"
                    title += f"Total: {total_training_steps}, Episode: {episodes_completed}, Reward: {episode_reward:.2f}\n"
                    
                    if current_option is not None:
                        title += f"Current Option: {current_option}, Option Changes: {option_changes}\n"
                    title += f"Epsilon: {epsilon:.4f}"
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
                success = info.get('goal_reached', False)
                print(f"  Episode {episodes_completed} completed after {episode_steps} steps")
                print(f"  Reward: {episode_reward:.2f}, Success: {success}")
                
                # Track metrics
                metric_data = metrics_tracker.get_episode_metrics(episode_reward, success, episode_steps)
                
                # Plot metrics periodically
                if total_training_steps % metrics_tracker.plot_interval == 0:
                    metrics_tracker.plot_metrics(total_training_steps)
                
                # Get new state
                state = handle_env_reset(env)
                state = preprocess_state(state)
                done = False
                episode_reward = 0
                episode_steps = 0
                episodes_completed += 1
                
                # Show reset state if rendering is enabled and not at end of training steps
                if render_training and main_ax is not None and step < num_train_steps:
                    maze_img = env.render(mode='rgb_array')
                    if maze_img is not None:
                        main_ax.clear()
                        main_ax.imshow(maze_img)
                        epsilon = agent._get_epsilon()
                        title = f"Training - Iteration {iteration}/{num_iterations}, New Episode {episodes_completed}\n"
                        title += f"Total Steps: {total_training_steps}, Epsilon: {epsilon:.4f}"
                        main_ax.set_title(title)
                        plt.draw()
                        plt.pause(0.01)
            
            # Break if we've reached our step limit
            steps_this_iteration += 1
            if steps_this_iteration >= num_train_steps:
                break
        
        # Save checkpoint periodically
        if iteration % 20 == 0 or iteration == num_iterations:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'model_iter{iteration}.pt')
            agent.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Calculate elapsed time and ETA
        elapsed_time = time.time() - start_time
        avg_time_per_iteration = elapsed_time / iteration
        remaining_iterations = num_iterations - iteration
        eta_seconds = avg_time_per_iteration * remaining_iterations
        
        print(f"Iteration {iteration} completed. Time elapsed: {elapsed_time:.1f}s, ETA: {eta_seconds:.1f}s")
    
    # Save final model
    final_checkpoint_path = os.path.join(config['checkpoint_dir'], 'final_model.pt')
    agent.save(final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    
    # Save final metrics and create plots
    metrics_tracker.plot_metrics(total_training_steps, final=True)
    metrics_tracker.save_metrics()
    print(f"Metrics saved to {metrics_tracker.results_dir}")
    
    # Close environment and plots
    env.close()
    if fig is not None:
        plt.close(fig)
    
    return agent


def get_config(maze_size=15, num_options=5, seed=None, fixed_layout=False):
    """Get configuration for the Complex Maze environment."""
    # Create timestamp for results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create configuration
    config = {
        # Environment params
        'maze_size': maze_size,
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
        
        # Rainbow DQN params (for DCEO)
        'noisy': False,
        'dueling': True,
        'double_dqn': True,
        'distributional': True,
        'num_atoms': 51,
        'v_min': -10.0,
        'v_max': 10.0,
        
        # DCEO-specific params
        'option_prob': 0.33,  # Prob of taking option level action
        'option_duration': 20,  # Typical option duration from paper
        'rep_dim': 64,  # Dimension of representation
        'log_transform': True,  # Apply log transform to reward
        'orthonormal': True,  # Orthonormal regularization
        'alpha_rep': 1.0,  # Weight for representation loss
        'alpha_main': 1.0,  # Weight for main task loss
        'alpha_option': 1.0,  # Weight for option loss
        
        # Path configuration
        'checkpoint_dir': os.path.join('maze_checkpoints', f'size{maze_size}_opt{num_options}'),
        'results_dir': os.path.join('maze_results', f'run_{timestamp}'),
        'checkpoint_every_n_steps': 10000
    }
    
    return config


def main():
    """Main function to run training or evaluation."""
    parser = argparse.ArgumentParser(description="Train or evaluate the fully online DCEO agent on Complex Maze")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps per iteration")
    parser.add_argument("--maze_size", type=int, default=15, help="Size of the maze (minimum 10 recommended)")
    parser.add_argument("--num_options", type=int, default=5, help="Number of options for DCEO")
    parser.add_argument("--render", action="store_true", help="Render visualization during training/evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to load checkpoint from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--fixed_layout", action="store_true", help="Use a fixed maze layout throughout training")
    
    args = parser.parse_args()
    
    # Ensure maze size is at least 10 to avoid errors in maze generation
    if args.maze_size < 10:
        print(f"Warning: Maze size {args.maze_size} is too small. Setting maze size to 10 (minimum recommended size).")
        args.maze_size = 10
    
    # Get configuration
    config = get_config(
        maze_size=args.maze_size,
        num_options=args.num_options,
        seed=args.seed,
        fixed_layout=args.fixed_layout
    )
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
    
    if args.mode == "train":
        print("Training fully online DCEO agent on Complex Maze")
        agent = train_agent(
            config=config,
            num_iterations=args.iterations,
            num_train_steps=args.steps,
            render_training=args.render
        )
    
    elif args.mode == "eval":
        print("Evaluating fully online DCEO agent on Complex Maze")
        
        # Create environment
        env = ComplexMazeEnv(
            maze_size=config['maze_size'],
            max_steps=config['maze_size'] * 4,
            use_fixed_seed=config.get('use_fixed_seed', False),
            fixed_seed=config.get('fixed_seed', None),
            use_fixed_layout=config.get('use_fixed_layout', False)
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
        
        # Set up visualization only if render is enabled
        fig = None
        axes = None
        if args.render:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Evaluate agent
        evaluate_agent(agent, env, num_episodes=10, fig=fig, axes=axes, render=args.render)
        
        # Close environment
        env.close()


if __name__ == "__main__":
    main()
