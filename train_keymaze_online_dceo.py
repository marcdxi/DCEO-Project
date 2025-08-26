"""
Training script for the fully online Rainbow DCEO agent on the Key-Door Maze environment.
This implements the training procedure for Algorithm 1 from Klissarov et al. (2023).
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

# Configure matplotlib for interactive mode
plt.ion()  # Enable interactive mode

# Import PyTorch online DCEO implementation
from pytorch_dceo_online import FullyOnlineDCEOAgent

# Import Key-Door Maze environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'key-maze'))
from key_door_maze_interactive import InteractiveKeyDoorMazeEnv

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess_state(state, shape=(84, 84)):
    """Preprocess the state from the environment for the agent.
    
    Args:
        state: State from Key-Door Maze environment (observation or tuple)
        shape: Target shape for network input
        
    Returns:
        Preprocessed state (C, H, W) normalized to [0, 1]
    """
    import numpy as np
    import cv2
    
    # Handle gym's new API which returns (obs, info) tuples
    if isinstance(state, tuple):
        state = state[0]  # Extract observation
    
    # For Key-Door Maze environment, state is a multi-channel grid
    # with shape (maze_size, maze_size, channels)
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
    # No need to normalize as values are already in [0, 1]
    rgb_image = rgb_image.transpose(2, 0, 1).astype(np.float32)
    
    return rgb_image


def handle_env_step(env, action):
    """Universal step function that works with both old and new Gym APIs.
    
    Args:
        env: The environment
        action: The action to take
        
    Returns:
        tuple: (next_state, reward, done, info)
    """
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
    """Universal reset function that works with both old and new Gym APIs.
    
    Args:
        env: The environment
        **kwargs: Additional arguments to pass to reset
    
    Returns:
        The initial state
    """
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


def evaluate_agent(agent, env, num_episodes=5, render=True):
    """Evaluate the agent on a number of episodes."""
    total_rewards = []
    total_steps = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = handle_env_reset(env)
        state = preprocess_state(state)
        
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.select_action(state_tensor, eval_mode=True)
            
            # Take action in environment
            next_state, reward, done, info = handle_env_step(env, action)
            next_state = preprocess_state(next_state)
            
            # Render if requested
            if render:
                env.render()
                time.sleep(0.01)  # Small delay for visualization
            
            # Update tracking variables
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Check for success
            if info.get('got_key', False):
                print(f"Episode {episode+1}: Got key at step {steps}")
            if info.get('opened_door', False):
                print(f"Episode {episode+1}: Opened door at step {steps}")
            if info.get('success', False):
                success_count += 1
                print(f"Episode {episode+1}: Success!")
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    # Print summary statistics
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = success_count / num_episodes
    
    print(f"\nEvaluation summary:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Success rate: {success_rate:.2f} ({success_count}/{num_episodes})")
    
    return avg_reward, success_rate


def train_agent(config, num_iterations=100, eval_frequency=10, num_train_steps=10000, render_training=False):
    """Train the fully online Rainbow DCEO agent on the Key-Door Maze environment."""
    
    # Create interactive environment
    env = InteractiveKeyDoorMazeEnv(
        maze_size=config['maze_size'],
        num_keys=config['num_keys'],
        max_steps=config['maze_size'] * 4
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
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # For tracking progress
    best_success_rate = 0.0
    total_training_steps = 0
    training_rewards = []
    eval_rewards = []
    success_rates = []
    
    start_time = time.time()
    
    for iteration in range(1, num_iterations + 1):
        # Training phase
        state = handle_env_reset(env)
        state = preprocess_state(state)
        done = False
        episode_reward = 0
        episode_steps = 0
        episodes_completed = 0
        
        print(f"\nIteration {iteration}/{num_iterations} - Training")
        
        for step in range(1, num_train_steps + 1):
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.select_action(state_tensor)
            
            # Take action in environment
            next_state, reward, done, info = handle_env_step(env, action)
            next_state = preprocess_state(next_state)
            
            # Render if requested
            if render_training and step % 10 == 0:  # Render every 10 steps to speed up training
                env.render()
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent - this is the fully online update
            agent.update()
            
            # Update tracking variables
            episode_reward += reward
            episode_steps += 1
            total_training_steps += 1
            state = next_state
            
            # Episode completed or truncated
            if done:
                training_rewards.append(episode_reward)
                
                # Print progress
                if episodes_completed % 10 == 0:
                    print(f"Episode {episodes_completed} - Reward: {episode_reward:.2f}, Steps: {episode_steps}")
                
                # Reset environment for next episode
                state = handle_env_reset(env)
                state = preprocess_state(state)
                done = False
                episode_reward = 0
                episode_steps = 0
                episodes_completed += 1
            
            # Save checkpoint periodically
            if total_training_steps % config['checkpoint_every_n_steps'] == 0:
                checkpoint_path = os.path.join(config['checkpoint_dir'], f"checkpoint_{total_training_steps}")
                os.makedirs(checkpoint_path, exist_ok=True)
                agent.save(checkpoint_path)
                print(f"Saved checkpoint at {checkpoint_path}")
        
        # Evaluation phase
        if iteration % eval_frequency == 0:
            print(f"\nIteration {iteration}/{num_iterations} - Evaluation")
            avg_reward, success_rate = evaluate_agent(agent, env, num_episodes=5, render=True)
            
            eval_rewards.append(avg_reward)
            success_rates.append(success_rate)
            
            # Save best model
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_model_path = os.path.join(config['checkpoint_dir'], "best_model")
                os.makedirs(best_model_path, exist_ok=True)
                agent.save(best_model_path)
                print(f"New best model with success rate {best_success_rate:.2f} saved at {best_model_path}")
        
        # Print time elapsed
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    # Final evaluation
    print("\nFinal Evaluation")
    avg_reward, success_rate = evaluate_agent(agent, env, num_episodes=10, render=True)
    
    # Save final model
    final_model_path = os.path.join(config['checkpoint_dir'], "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    agent.save(final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Total training steps: {total_training_steps}")
    print(f"Best success rate: {best_success_rate:.2f}")
    print(f"Final success rate: {success_rate:.2f}")
    print(f"Final average reward: {avg_reward:.2f}")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    
    # Close environment
    env.close()
    
    return agent


def get_config(maze_size=10, num_keys=1, num_options=5):
    """Get configuration for the Key-Door Maze environment."""
    # Default configuration for Key-Door Maze
    config = {
        # Environment settings
        "env_name": "KeyDoorMaze",
        "input_shape": [3, 84, 84],
        "num_actions": 4,
        "maze_size": maze_size,
        "num_keys": num_keys,
        
        # Rainbow settings
        "buffer_size": 100000,  # Reduced for faster training on Key-Door Maze
        "batch_size": 32,
        "gamma": 0.99,
        "update_horizon": 3,
        "min_replay_history": 10000,  # Reduced for faster learning on Key-Door Maze
        "update_period": 4,
        "target_update_period": 8000,
        "epsilon_train": 0.01,
        "epsilon_eval": 0.001,
        "epsilon_decay_period": 250000,
        "learning_rate": 0.0001,  # Adjusted for faster learning
        
        # Rainbow features
        "noisy": True,
        "dueling": True,
        "double_dqn": True,
        "distributional": True,
        "num_atoms": 51,
        "v_min": -10,
        "v_max": 10,
        
        # DCEO settings
        "num_options": num_options,
        "option_prob": 0.9,
        "option_duration": 10,
        "rep_dim": 20,
        "log_transform": True,
        "orthonormal": True,  # Apply orthonormalization to representations
        "alpha_rep": 1.0,     # Coefficient for representation loss
        "alpha_main": 1.0,    # Coefficient for main policy loss
        "alpha_option": 1.0,  # Coefficient for option policy loss
        
        # Training settings
        "checkpoint_every_n_steps": 50000,
        "checkpoint_dir": "./online_dceo_checkpoints"
    }
    
    return config


def main():
    """Main function to run training or evaluation."""
    parser = argparse.ArgumentParser(description="Train or evaluate the fully online DCEO agent on KeyDoorMaze")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode: train or eval")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--maze_size", type=int, default=10, help="Size of the maze (minimum 8 recommended)")
    parser.add_argument("--num_keys", type=int, default=1, help="Number of keys in the maze")
    parser.add_argument("--num_options", type=int, default=5, help="Number of options for DCEO")
    parser.add_argument("--render", action="store_true", help="Render training episodes")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to load checkpoint from")
    
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
        num_options=args.num_options
    )
    
    if args.mode == "train":
        print("Training fully online DCEO agent on KeyDoorMaze")
        agent = train_agent(
            config=config,
            num_iterations=args.iterations,
            render_training=args.render
        )
    
    elif args.mode == "eval":
        print("Evaluating fully online DCEO agent on KeyDoorMaze")
        
        # Create interactive environment
        env = InteractiveKeyDoorMazeEnv(
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
        
        # Evaluate agent
        evaluate_agent(agent, env, num_episodes=10, render=True)
        
        # Close environment
        env.close()


if __name__ == "__main__":
    main()
