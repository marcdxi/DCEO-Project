"""
Training script for the fully online Rainbow DCEO agent on the Mountain Car environment.
This implementation adapts DCEO to a continuous state space environment.
"""

import numpy as np
import torch
import time
import os
import sys
import argparse
import json
import pickle
import datetime
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [12, 8]  # Larger figure size
matplotlib.rcParams['figure.dpi'] = 100  # Higher DPI
plt.ion()  # Enable interactive mode globally

# Setup minimal logging
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('mountain_car_dceo')
from matplotlib.patches import Rectangle
from IPython.display import clear_output
import gym

# Import PyTorch online DCEO implementation
from pytorch_dceo_online import FullyOnlineDCEOAgent

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess_state(state):
    """Preprocess the state from the Mountain Car environment for the agent.
    Transforms the raw state (position, velocity) into a 3-channel image representation
    to match the expected input format of the DCEO agent."""
    
    # Handle gym's new API which returns (obs, info) tuples
    if isinstance(state, tuple):
        state = state[0]  # Extract observation
    
    # Convert raw MountainCar state to a visual representation
    # state[0] is position (-1.2 to 0.6)
    # state[1] is velocity (-0.07 to 0.07)
    
    # Create a simple 84x84 visual representation
    visual_state = np.zeros((3, 84, 84), dtype=np.float32)
    
    # Normalize position to 0-1 range
    pos_normalized = (state[0] + 1.2) / 1.8  # Map [-1.2, 0.6] to [0, 1]
    
    # Normalize velocity to 0-1 range
    vel_normalized = (state[1] + 0.07) / 0.14  # Map [-0.07, 0.07] to [0, 1]
    
    # Create visual representation:
    # Channel 0: Position (horizontal bar)
    pos_pixel = int(pos_normalized * 83)
    visual_state[0, 41:43, :] = 0.3  # Draw the "track"
    visual_state[0, 35:48, pos_pixel-2:pos_pixel+3] = 1.0  # Car position
    
    # Channel 1: Velocity (color intensity)
    visual_state[1] = vel_normalized * np.ones((84, 84))
    
    # Channel 2: Goal location marker
    goal_pixel = int((0.6 + 1.2) / 1.8 * 83)
    visual_state[2, 30:55, goal_pixel-4:goal_pixel+4] = 1.0
    
    return visual_state


def handle_env_step(env, action, max_position=None):
    """Universal step function that works with both old and new Gym APIs.
    Also implements custom reward structure for Mountain Car.
    
    Args:
        env: The environment
        action: Action to take
        max_position: Current maximum position reached (for reward shaping)
        
    Returns:
        next_state, reward, done, info
    """
    step_result = env.step(action)
    
    # Handle different step() return formats
    next_state = None
    reward = None
    done = None
    info = {}
    
    if isinstance(step_result, tuple):
        if len(step_result) == 5:  # New Gymnasium API: (next_state, reward, terminated, truncated, info)
            next_state, original_reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  # Old Gym API: (next_state, reward, done, info)
            next_state, original_reward, done, info = step_result
    else:
        raise ValueError(f"Unexpected step result type: {type(step_result)}")
    
    # Apply a more moderate reward structure as recommended
    # 1. Base reward: -0.01 per step penalty
    reward = -0.01  # Small step penalty
    
    # 2. Goal achievement: +1.0 when the car reaches the flag
    if next_state[0] >= 0.5:
        reward += 1.0
        info['goal_reached'] = True
    else:
        info['goal_reached'] = False
    
    # 3. More moderate height-based shaping
    if max_position is not None and next_state[0] > max_position:
        # Scale reward by how much higher the car got, but keep it smaller
        height_diff = next_state[0] - max_position
        height_bonus = height_diff * 2.0  # More moderate scaling
        reward += height_bonus
        info['new_max_height'] = True
        info['height_bonus'] = height_bonus
    else:
        info['new_max_height'] = False
        info['height_bonus'] = 0.0
    
    # 4. Smaller velocity bonus only when moving uphill toward the goal
    # (moving right with positive velocity or moving left with negative velocity)
    # This encourages building momentum in the useful direction
    if (next_state[0] > 0 and next_state[1] > 0) or (next_state[0] < 0 and next_state[1] < 0):
        velocity_bonus = abs(next_state[1]) * 0.05  # Much smaller velocity bonus
        reward += velocity_bonus
        info['velocity_bonus'] = velocity_bonus
    else:
        info['velocity_bonus'] = 0.0
    
    # Store original reward for reference
    info['original_reward'] = original_reward
    
    return next_state, reward, done, info


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


def save_training_metrics(metrics, iteration, output_dir="./results"):
    """Save training metrics to the results folder.
    
    Args:
        metrics: Dictionary of metrics to save
        iteration: Current iteration number
        output_dir: Base output directory
    """
    # Create timestamp for this training run
    if not hasattr(save_training_metrics, "timestamp"):
        save_training_metrics.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    run_dir = os.path.join(output_dir, f"run_{save_training_metrics.timestamp}")
    metrics_dir = os.path.join(run_dir, "metrics")
    plots_dir = os.path.join(run_dir, "plots")
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = os.path.join(metrics_dir, f"metrics_iter_{iteration}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save learning curves
    if 'learning_curves' in metrics:
        curves = metrics['learning_curves']
        
        # Plot rewards
        plt.figure(figsize=(12, 8))
        iterations = list(range(1, len(curves['rewards']) + 1))
        plt.plot(iterations, curves['rewards'], 'o-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Reward Progress')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'reward_curve.png'))
        plt.close()
        
        # Plot success rates if available
        if 'success_rates' in curves:
            plt.figure(figsize=(12, 8))
            plt.plot(iterations, curves['success_rates'], 'o-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Success Rate')
            plt.title('Success Rate Progress')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'success_rate_curve.png'))
            plt.close()
    
    # Save option usage statistics if available
    if 'option_usage' in metrics:
        option_file = os.path.join(metrics_dir, f"option_usage_iter_{iteration}.pkl")
        with open(option_file, 'wb') as f:
            pickle.dump(metrics['option_usage'], f)
    
    # Save configuration
    if 'config' in metrics:
        config_file = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_file):  # Only save once
            with open(config_file, 'w') as f:
                json.dump(metrics['config'], f, indent=2)
    
    logger.info(f"Metrics saved to {run_dir}")
    return run_dir

def evaluate_agent(agent, env, num_episodes=5, fig=None, axes=None, plot_metrics=None):
    """Evaluate the agent and visualize performance."""
    print(f"\nEvaluating agent over {num_episodes} episodes...")
    total_rewards = []
    total_steps = []
    success_count = 0
    
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
    plt.show(block=False)
    try:
        # Make window appear on top
        fig.canvas.manager.window.attributes('-topmost', 1)  
        fig.canvas.manager.window.attributes('-topmost', 0)  
        main_ax.figure.canvas.draw()
        main_ax.figure.canvas.flush_events()
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    for episode in range(num_episodes):
        state = handle_env_reset(env)
        state = preprocess_state(state)
        
        done = False
        episode_reward = 0
        steps = 0
        max_height = -1.2  # Track maximum height reached
        max_position = -1.2  # Track maximum position for reward shaping
        
        while not done:
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.select_action(state_tensor, eval_mode=True)
            
            # Take action in environment
            next_state, reward, done, info = handle_env_step(env, action, max_position)
            
            # Track maximum height and position
            max_height = max(max_height, next_state[0])
            max_position = max(max_position, next_state[0])
            
            # Add reward info to title
            if info.get('new_max_height', False):
                print(f"  New max height: {max_position:.2f}, received height bonus")
            
            next_state = preprocess_state(next_state)
            
            # Update tracking variables
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Check for success (reaching the goal flag)
            goal_reached = max_height >= 0.5
            
            # Get Mountain Car image
            if hasattr(env, 'render'):
                try:
                    render_mode = 'rgb_array'
                    mc_img = env.render(mode=render_mode)
                except Exception:
                    try:
                        # Try new Gymnasium API
                        mc_img = env.render()
                    except Exception:
                        mc_img = None
                        
                if mc_img is not None:
                    # Draw Mountain Car environment
                    main_ax.clear()
                    main_ax.imshow(mc_img)
                    
                    # Add title with information
                    title = f"Evaluation - Episode {episode+1}/{num_episodes}, Step {steps}\n"
                    title += f"Reward: {episode_reward:.2f}, Max Height: {max_height:.2f}"
                    main_ax.set_title(title)
                    
                    # Update plot
                    try:
                        plt.draw()
                        plt.pause(0.01)
                    except Exception as e:
                        print(f"Error updating visualization: {e}")
            
            # Log success
            if goal_reached and not done:
                success_count += 1
                print(f"Episode {episode+1}: Success! Reached goal at step {steps}")
                done = True  # End episode on success
            
            # Limit evaluation episode length
            if steps >= 1000:
                done = True
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}, Max Height = {max_height:.2f}")
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    
    # Calculate metrics
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = success_count / num_episodes
    
    print(f"\nEvaluation Results (over {num_episodes} episodes):")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{num_episodes})")
    
    plot_metrics['rewards'].append(avg_reward)
    plot_metrics['success_rates'].append(success_rate)
    
    return avg_reward, success_rate, fig, axes, plot_metrics


def train_agent(config, num_iterations=100, eval_frequency=10, num_train_steps=10000, render_training=False):
    """Train the fully online Rainbow DCEO agent on the Mountain Car environment.
    
    Args:
        config: Agent configuration dictionary
        num_iterations: Number of training iterations
        eval_frequency: How often to evaluate the agent
        num_train_steps: Number of environment steps per training iteration
        render_training: Whether to render the environment during training
    """
    print("\nInitializing Mountain Car environment...")
    
    try:
        # Try importing from gymnasium first (newer API)
        import gymnasium as gym
        env = gym.make('MountainCar-v0', render_mode='rgb_array')
        print("Using Gymnasium API")
    except (ImportError, gym.error.Error):
        # Fall back to older gym API
        import gym
        env = gym.make('MountainCar-v0')
        print("Using OpenAI Gym API")
    
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Get input shape from preprocessed state
    sample_state = handle_env_reset(env)
    preprocessed_shape = preprocess_state(sample_state).shape
    print(f"Preprocessed state shape: {preprocessed_shape}")
    
    # Create agent
    print(f"Creating DCEO agent with preprocessed shape {preprocessed_shape}")
    agent = FullyOnlineDCEOAgent(
        input_shape=preprocessed_shape,
        num_actions=env.action_space.n,
        **config
    )
    
    # Setup for visualization and metrics tracking
    fig = None
    axes = None
    plot_metrics = {'rewards': [], 'success_rates': [], 'iterations': []}
    
    # Metrics collection
    all_metrics = {
        'config': config,
        'learning_curves': {
            'rewards': [],
            'success_rates': [],
            'max_heights': []
        },
        'option_usage': {},
        'environment_info': {
            'observation_space': str(env.observation_space),
            'action_space': str(env.action_space)
        }
    }
    
    # Train for specified number of iterations
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}/{num_iterations} - Training")
        print(f"{'='*50}")
        
        # Reset environment
        state = handle_env_reset(env)
        state = preprocess_state(state)
        
        done = False
        episode_reward = 0
        episode_steps = 0
        max_position = -1.2  # Track maximum position for reward shaping
        
        # Variables for tracking options
        last_option = None
        option_starts = {}
        option_durations = {}
        
        # Training loop
        for step in range(1, num_train_steps + 1):
            # Get current option
            current_option = agent.cur_opt
            if last_option is not None and current_option != last_option:
                # Option terminated, record duration
                if last_option in option_durations:
                    option_durations[last_option].append(step - option_starts[last_option])
                else:
                    option_durations[last_option] = [step - option_starts[last_option]]
            
            if current_option is not None and (last_option is None or current_option != last_option):
                # New option started
                option_starts[current_option] = step
            
            last_option = current_option
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Select action
            action = agent.select_action(state_tensor)
            
            # Take action in environment
            next_state, reward, done, info = handle_env_step(env, action, max_position)
            
            # Update maximum position
            max_position = max(max_position, next_state[0])
            
            # Report height bonus
            if info.get('new_max_height', False) and step % 100 == 0:
                print(f"  New max height: {max_position:.2f}")
                
            next_state = preprocess_state(next_state)
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            
            # Move to next state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Optionally render during training
            if render_training and step % 10 == 0:
                if hasattr(env, 'render'):
                    try:
                        render_mode = 'rgb_array'
                        mc_img = env.render(mode=render_mode)
                    except Exception:
                        try:
                            # Try new Gymnasium API
                            mc_img = env.render()
                        except Exception:
                            mc_img = None
                    
                    if mc_img is not None and fig is not None and axes is not None:
                        main_ax = axes[0]
                        main_ax.clear()
                        main_ax.imshow(mc_img)
                        main_ax.set_title(f"Training - Step {step}")
                        try:
                            plt.draw()
                            plt.pause(0.01)
                        except Exception as e:
                            print(f"Error in visualization: {e}")
            
            # Check if episode is done
            if done:
                print(f"Training episode finished: {episode_steps} steps, reward {episode_reward:.2f}")
                
                # Reset for next episode
                state = handle_env_reset(env)
                state = preprocess_state(state)
                done = False
                episode_reward = 0
                episode_steps = 0
            
            # Display progress
            if step % 1000 == 0:
                print(f"Step {step}/{num_train_steps}")
                    
                # Report option statistics if available
                print("\nOption usage statistics:")
                for opt_id, stats in option_usage.items():
                    print(f"  Option {opt_id}: used {stats['count']} times, avg duration: {stats['avg_duration']:.2f} steps")
                    
            # Store option usage statistics in metrics
            all_metrics['option_usage'][iteration] = option_usage.copy()
                
        # Evaluate agent performance
        if iteration % eval_frequency == 0 or iteration == num_iterations:
            print(f"\n{'='*50}")
            print(f"Iteration {iteration}/{num_iterations} - Evaluation")
            print(f"{'='*50}\n")
                
            avg_reward, success_rate, fig, axes, plot_metrics = evaluate_agent(
                agent, env, num_episodes=5, fig=fig, axes=axes, plot_metrics=plot_metrics)
                
            # Extract max height from evaluation
            max_height = max([ep_info.get('max_height', -1.2) for ep_info in agent.episode_info[-5:]]) if hasattr(agent, 'episode_info') else -1.2
                
            # Store evaluation metrics
            all_metrics['learning_curves']['rewards'].append(avg_reward)
            all_metrics['learning_curves']['success_rates'].append(success_rate)
            all_metrics['learning_curves']['max_heights'].append(max_height)
                
            # Save metrics to results folder
            save_training_metrics(all_metrics, iteration)
                
            # Save checkpoint
            print(f"\nSaving checkpoint at iteration {iteration}")
            checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
                
            checkpoint_path = os.path.join(checkpoint_dir, f'mountain_car_dceo_iter_{iteration}.pt')
            torch.save({
                'iteration': iteration,
                'agent_state_dict': agent.state_dict(),
                'metrics': all_metrics,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
                
    print("\nTraining completed!")
        
    # Close environment
    env.close()
    
    # Save final model with metrics
    final_path = os.path.join(checkpoint_dir, 'mountain_car_dceo_final.pt')
    torch.save({
        'iteration': num_iterations,
        'agent_state_dict': agent.state_dict(),
        'metrics': all_metrics,
        'config': config
    }, final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # Save final metrics
    final_metrics_dir = save_training_metrics(all_metrics, num_iterations, output_dir="./mountain_car_dceo/results")
    print(f"Final training metrics saved to: {final_metrics_dir}")
    
    # Generate additional analysis plots
    if all_metrics['learning_curves']['rewards']:
        # Create a results directory specifically for Mountain Car DCEO
        mc_results_dir = os.path.join("./mountain_car_dceo/results")
        os.makedirs(mc_results_dir, exist_ok=True)
        
        # Plot learning curves
        plt.figure(figsize=(15, 12))
        
        # Rewards
        plt.subplot(3, 1, 1)
        iterations = list(range(1, len(all_metrics['learning_curves']['rewards']) + 1))
        plt.plot(iterations, all_metrics['learning_curves']['rewards'], 'o-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Reward Progress')
        plt.grid(True)
        
        # Success Rates
        plt.subplot(3, 1, 2)
        plt.plot(iterations, all_metrics['learning_curves']['success_rates'], 'o-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Progress')
        plt.grid(True)
        
        # Max Heights
        plt.subplot(3, 1, 3)
        plt.plot(iterations, all_metrics['learning_curves']['max_heights'], 'o-', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Goal Position')
        plt.xlabel('Iteration')
        plt.ylabel('Max Height')
        plt.title('Maximum Height Progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(mc_results_dir, 'learning_progress.png'))
        plt.close()
        print(f"Overall learning progress plot saved to: {mc_results_dir}/learning_progress.png")
    
    return agent, plot_metrics


def get_config(num_options=5):
    """Get configuration for Mountain Car environment.
    
    Args:
        num_options: Number of options to learn
        
    Returns:
        config: Configuration dictionary for agent
    """
    config = {
        # Rainbow parameters
        'buffer_size': 100000,  # Smaller buffer for faster updates
        'batch_size': 128,      # Larger batch size for more stable learning
        'gamma': 0.99,
        'update_horizon': 3,
        'min_replay_history': 200,  # Start learning much sooner
        'update_period': 2,        # Update more frequently
        'target_update_period': 800,  # More frequent target network updates
        'epsilon_train': 0.3,       # Higher exploration rate for better coverage
        'epsilon_eval': 0.05,
        'epsilon_decay_period': 25000,  # Slower decay for sustained exploration
        'learning_rate': 0.001,     # Increased learning rate for faster updates
        'noisy': True,
        'dueling': True,
        'double_dqn': True,
        'distributional': True,
        'num_atoms': 51,
        'v_min': -10,
        'v_max': 10,
        
        # DCEO parameters
        'num_options': num_options,
        'option_prob': 0.95,         # High probability of using options
        'option_duration': 10,        # Longer option duration for better temporal abstraction
        'rep_dim': 32,                # Larger representation dimension for more expressive features
        'log_transform': True,        # Apply log transform to rewards
        'orthonormal': True,          # Apply orthonormalization to representations
        'alpha_rep': 1.0,             # Coefficient for representation loss
        'alpha_main': 1.0,            # Coefficient for main policy loss
        'alpha_option': 1.5           # Higher emphasis on option learning for better option discovery
    }
    
    return config


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train fully online DCEO agent on Mountain Car')
    parser.add_argument('--iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--eval_freq', type=int, default=5, help='Evaluation frequency (iterations)')
    parser.add_argument('--train_steps', type=int, default=10000, help='Steps per training iteration')
    parser.add_argument('--num_options', type=int, default=5, help='Number of options to learn')
    parser.add_argument('--render', action='store_true', help='Render training')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Fully Online DCEO for Mountain Car")
    print("=" * 60)
    
    # Get config
    config = get_config(num_options=args.num_options)
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train agent
    agent, metrics = train_agent(
        config,
        num_iterations=args.iterations,
        eval_frequency=args.eval_freq,
        num_train_steps=args.train_steps,
        render_training=args.render
    )
    
    # Final evaluation
    try:
        import gymnasium as gym
        env = gym.make('MountainCar-v0', render_mode='rgb_array')
    except (ImportError, gym.error.Error):
        import gym
        env = gym.make('MountainCar-v0')
    
    print("\nFinal Evaluation:")
    evaluate_agent(agent, env, num_episodes=10)
    
    env.close()


if __name__ == "__main__":
    main()
