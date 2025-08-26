"""
Visualization script for DCEO Mountain Car options.
This script loads checkpoints and visualizes the learned option policies, termination conditions,
and state visitation patterns to reveal how the agent is using temporal abstraction.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import argparse
import gymnasium as gym
from tqdm import tqdm

# Add import paths
import sys
sys.path.append('.')
sys.path.append('..')

from mountain_car_dceo.train_mountain_car_dceo import preprocess_state, get_config, handle_env_reset
from pytorch_dceo_online import FullyOnlineDCEOAgent

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_environment():
    """Create the Mountain Car environment."""
    try:
        # Try importing from gymnasium first (newer API)
        import gymnasium as gym
        env = gym.make('MountainCar-v0', render_mode='rgb_array')
        print("Using Gymnasium API")
    except (ImportError, Exception):
        # Fall back to older gym API
        import gym
        env = gym.make('MountainCar-v0')
        print("Using OpenAI Gym API")
    
    return env

def load_checkpoint(checkpoint_path):
    """Load a checkpoint file safely."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Successfully loaded checkpoint")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Trying to load with different approach...")
        checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
        print("Successfully loaded checkpoint with alternative method")
    
    # Extract iteration number
    if 'iteration' in checkpoint:
        iteration = checkpoint['iteration']
    else:
        # Get from filename
        import re
        match = re.search(r'iter_(\d+)', checkpoint_path)
        if match:
            iteration = int(match.group(1))
        else:
            iteration = 0
    
    # If final checkpoint, use a high iteration number
    if 'final' in checkpoint_path:
        print("This is the final checkpoint")
    
    return checkpoint, iteration

def initialize_agent(env):
    """Initialize a DCEO agent with default config."""
    # Get config
    config = get_config()
    
    # Get input shape from preprocessed state
    sample_state = handle_env_reset(env)
    preprocessed_shape = preprocess_state(sample_state).shape
    print(f"Preprocessed state shape: {preprocessed_shape}")
    
    # Create agent
    agent = FullyOnlineDCEOAgent(
        input_shape=preprocessed_shape,
        num_actions=env.action_space.n,
        **config
    )
    
    return agent

def load_agent_weights(agent, checkpoint):
    """Load agent weights from checkpoint."""
    # Try different possible state dict keys
    state_dict_keys = ['agent_state_dict', 'state_dict', 'model_state_dict', 'network_state_dict']
    
    for key in state_dict_keys:
        if key in checkpoint:
            print(f"Loading state dict from '{key}'")
            try:
                agent.load_state_dict(checkpoint[key])
                return True
            except Exception as e:
                print(f"Error loading from '{key}': {e}")
    
    print("WARNING: Could not find or load state dict in checkpoint")
    return False

def compute_option_policies(agent, pos_range, vel_range):
    """Compute option policies across the state space."""
    num_options = agent.num_options
    option_policies = np.zeros((num_options, len(pos_range), len(vel_range)))
    termination_probs = np.zeros((num_options, len(pos_range), len(vel_range)))
    option_values = np.zeros((num_options, len(pos_range), len(vel_range)))
    
    for i, pos in enumerate(tqdm(pos_range, desc="Processing positions")):
        for j, vel in enumerate(vel_range):
            state = np.array([pos, vel], dtype=np.float32)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Preprocess state
            processed_state = preprocess_state(state)
            processed_state = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Extract policy and termination probability for each option
                for opt in range(num_options):
                    # Get option value network's output for this state
                    q_values = agent.rainbow_net.option_value_networks[opt](processed_state)
                    
                    # Distributional RL uses a distribution over values, so we need to aggregate
                    # Mean over atoms (value distribution) dimension
                    mean_q_values = q_values.mean(dim=2)
                    
                    # Sum over batch dimension (only one sample)
                    summed_q_values = mean_q_values.sum(dim=0)
                    
                    # Get best action and its value
                    action = torch.argmax(summed_q_values).item()
                    option_value = summed_q_values.max().item()
                    
                    # Get termination probability
                    term_logits = agent.rainbow_net.termination_networks[opt](processed_state)
                    term_prob = torch.sigmoid(term_logits).item()
                    
                    # Store results
                    option_policies[opt, i, j] = action
                    termination_probs[opt, i, j] = term_prob
                    option_values[opt, i, j] = option_value
    
    return option_policies, termination_probs, option_values

def plot_option_policies(option_policies, termination_probs, option_values, pos_range, vel_range, iteration, save_dir):
    """Plot option policies and termination probabilities."""
    num_options = option_policies.shape[0]
    
    # Create a custom colormap for policies (Left=0: Blue, No-op=1: Green, Right=2: Red)
    cmap = LinearSegmentedColormap.from_list('action_cmap', ['blue', 'green', 'red'], N=3)
    
    # Create directory for saving
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot policies (one figure per option)
    for opt in range(num_options):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot policy
        policy_img = axs[0].imshow(option_policies[opt].T, origin='lower', aspect='auto', 
                               extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
                               cmap=cmap, vmin=0, vmax=2)
        axs[0].set_title(f'Option {opt+1} Policy\n(Blue=Left, Green=No-op, Red=Right)')
        axs[0].set_xlabel('Position')
        axs[0].set_ylabel('Velocity')
        fig.colorbar(policy_img, ax=axs[0], ticks=[0, 1, 2], 
                    label='Action')
        
        # Plot termination probability
        term_img = axs[1].imshow(termination_probs[opt].T, origin='lower', aspect='auto',
                             extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
                             cmap='viridis', vmin=0, vmax=1)
        axs[1].set_title(f'Option {opt+1} Termination Probability')
        axs[1].set_xlabel('Position')
        axs[1].set_ylabel('Velocity')
        fig.colorbar(term_img, ax=axs[1], label='Probability')
        
        # Plot option value
        val_img = axs[2].imshow(option_values[opt].T, origin='lower', aspect='auto',
                           extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
                           cmap='plasma')
        axs[2].set_title(f'Option {opt+1} Value Function')
        axs[2].set_xlabel('Position')
        axs[2].set_ylabel('Velocity')
        fig.colorbar(val_img, ax=axs[2], label='Value')
        
        # Add goal line
        for ax in axs:
            ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.8, label='Goal')
            
        plt.suptitle(f'Option {opt+1} Analysis - Iteration {iteration}', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, f'option_{opt+1}_iter_{iteration}.png')
        plt.savefig(save_path)
        plt.close()
    
    # Create a combined policy visualization
    fig, axs = plt.subplots(2, num_options, figsize=(4*num_options, 8))
    
    # First row: Option policies
    for opt in range(num_options):
        policy_img = axs[0, opt].imshow(option_policies[opt].T, origin='lower', aspect='auto',
                                    extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
                                    cmap=cmap, vmin=0, vmax=2)
        axs[0, opt].set_title(f'Option {opt+1} Policy')
        axs[0, opt].set_xlabel('Position')
        if opt == 0:
            axs[0, opt].set_ylabel('Velocity')
        axs[0, opt].axvline(x=0.5, color='white', linestyle='--', alpha=0.8)
    
    # Common colorbar for policies
    cbar_policy = fig.colorbar(policy_img, ax=axs[0, :], location='right', shrink=0.8)
    cbar_policy.set_ticks([0, 1, 2])
    cbar_policy.set_ticklabels(['Left', 'No-op', 'Right'])
    
    # Second row: Termination probabilities
    for opt in range(num_options):
        term_img = axs[1, opt].imshow(termination_probs[opt].T, origin='lower', aspect='auto',
                                  extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
                                  cmap='viridis', vmin=0, vmax=1)
        axs[1, opt].set_title(f'Option {opt+1} Termination')
        axs[1, opt].set_xlabel('Position')
        if opt == 0:
            axs[1, opt].set_ylabel('Velocity')
        axs[1, opt].axvline(x=0.5, color='white', linestyle='--', alpha=0.8)
    
    # Common colorbar for termination probabilities
    fig.colorbar(term_img, ax=axs[1, :], location='right', shrink=0.8)
    
    plt.suptitle(f'All Options Comparison - Iteration {iteration}', fontsize=16)
    plt.tight_layout()
    
    # Save combined figure
    save_path = os.path.join(save_dir, f'all_options_iter_{iteration}.png')
    plt.savefig(save_path)
    plt.close()

def simulate_option_trajectories(agent, env, num_episodes=20, max_steps=200):
    """Simulate agent behavior to record option usage trajectories."""
    option_trajectories = []
    option_durations = [[] for _ in range(agent.num_options)]
    option_counts = np.zeros(agent.num_options)
    
    for episode in range(num_episodes):
        print(f"Simulating episode {episode+1}/{num_episodes}")
        state, _ = env.reset()
        done = False
        steps = 0
        
        trajectory = []
        current_option = None
        option_step_count = 0
        
        while not done and steps < max_steps:
            # Process state
            state_tensor = preprocess_state(state)
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get action and option info
            with torch.no_grad():
                action, option_info = agent.act(state_tensor, False)
            
            # Track option usage
            option_id = option_info.get('option_id', None)
            
            # If option changed, record the duration of the previous option
            if option_id != current_option and current_option is not None:
                option_durations[current_option].append(option_step_count)
            
            # Update current option and step count
            if option_id != current_option:
                current_option = option_id
                option_step_count = 1
                if current_option is not None:
                    option_counts[current_option] += 1
            else:
                option_step_count += 1
            
            # Store state, action and option
            trajectory.append({
                'position': state[0],
                'velocity': state[1],
                'option': option_id,
                'action': action
            })
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            steps += 1
        
        # Add final option duration
        if current_option is not None:
            option_durations[current_option].append(option_step_count)
        
        option_trajectories.append(trajectory)
    
    # Calculate average durations
    avg_durations = [np.mean(durations) if durations else 0 for durations in option_durations]
    
    return option_trajectories, option_counts, avg_durations

def plot_option_trajectories(option_trajectories, option_counts, avg_durations, iteration, save_dir, pos_range, vel_range):
    """Plot the trajectories taken by each option."""
    num_options = len(avg_durations)
    
    # Create a grid to count state visitations by option
    pos_bins = np.linspace(pos_range[0], pos_range[-1], 50)
    vel_bins = np.linspace(vel_range[0], vel_range[-1], 50)
    
    option_visitation = np.zeros((num_options, len(pos_bins)-1, len(vel_bins)-1))
    option_actions = [[] for _ in range(num_options)]
    
    # Count state visitations for each option
    for trajectory in option_trajectories:
        for step in trajectory:
            option_id = step['option']
            if option_id is not None:
                # Find the bin for this state
                pos_idx = np.digitize(step['position'], pos_bins) - 1
                vel_idx = np.digitize(step['velocity'], vel_bins) - 1
                
                if 0 <= pos_idx < len(pos_bins)-1 and 0 <= vel_idx < len(vel_bins)-1:
                    option_visitation[option_id, pos_idx, vel_idx] += 1
                    option_actions[option_id].append(step['action'])
    
    # Create figure for option visitation
    fig, axs = plt.subplots(1, num_options, figsize=(4*num_options, 4))
    
    for opt in range(num_options):
        # Normalize visitation counts for better visualization
        normed_visitation = option_visitation[opt] / (option_visitation[opt].max() + 1e-10)
        
        im = axs[opt].imshow(normed_visitation.T, origin='lower', aspect='auto',
                         extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
                         cmap='viridis')
        
        # Calculate most frequent action
        actions = option_actions[opt]
        if actions:
            action_counts = np.bincount(actions, minlength=3)
            most_frequent = np.argmax(action_counts)
            action_names = ['Left', 'No-op', 'Right']
            action_str = action_names[most_frequent]
        else:
            action_str = "N/A"
        
        axs[opt].set_title(f'Option {opt+1}\nCount: {int(option_counts[opt])} | Avg Duration: {avg_durations[opt]:.1f}\nMain Action: {action_str}')
        axs[opt].set_xlabel('Position')
        if opt == 0:
            axs[opt].set_ylabel('Velocity')
        axs[opt].axvline(x=0.5, color='white', linestyle='--', alpha=0.8)
    
    plt.suptitle(f'Option Visitation Patterns - Iteration {iteration}', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'option_trajectories_iter_{iteration}.png')
    plt.savefig(save_path)
    plt.close()

def visualize_checkpoint(checkpoint_path, save_dir='./mountain_car_dceo/results/option_visualizations'):
    """Visualize option policies from a checkpoint."""
    # Create environment
    env = create_environment()
    
    # Load checkpoint
    checkpoint, iteration = load_checkpoint(checkpoint_path)
    
    # Initialize agent
    agent = initialize_agent(env)
    
    # Load weights
    if not load_agent_weights(agent, checkpoint):
        print("WARNING: Using randomly initialized weights!")
    
    # Define state space ranges
    pos_range = np.linspace(-1.2, 0.6, 50)
    vel_range = np.linspace(-0.07, 0.07, 50)
    
    # Create directory for this iteration
    iteration_dir = os.path.join(save_dir, f'iteration_{iteration}')
    os.makedirs(iteration_dir, exist_ok=True)
    
    # Compute option policies
    print("Computing option policies across state space...")
    option_policies, termination_probs, option_values = compute_option_policies(
        agent, pos_range, vel_range)
    
    # Plot option policies
    print("Plotting option policies...")
    plot_option_policies(option_policies, termination_probs, option_values, 
                         pos_range, vel_range, iteration, iteration_dir)
    
    # Simulate option trajectories
    print("Simulating agent behavior to record option trajectories...")
    option_trajectories, option_counts, avg_durations = simulate_option_trajectories(
        agent, env, num_episodes=10, max_steps=200)
    
    # Plot option trajectories
    print("Plotting option trajectory patterns...")
    plot_option_trajectories(option_trajectories, option_counts, avg_durations, 
                            iteration, iteration_dir, [pos_range[0], pos_range[-1]], 
                            [vel_range[0], vel_range[-1]])
    
    print(f"Visualization complete for iteration {iteration}")
    return iteration_dir

def main():
    parser = argparse.ArgumentParser(description='Visualize DCEO option policies')
    parser.add_argument('--checkpoint_dir', type=str, default='./mountain_car_dceo/checkpoints',
                        help='Directory containing checkpoint files')
    parser.add_argument('--save_dir', type=str, default='./mountain_car_dceo/results/option_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--checkpoint', type=str, 
                        help='Specific checkpoint to visualize (default: use final/latest checkpoint)')
    parser.add_argument('--visualize_all', action='store_true',
                        help='Visualize all checkpoints in the directory')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.checkpoint:
        # Visualize specific checkpoint
        visualize_checkpoint(args.checkpoint, args.save_dir)
    elif args.visualize_all:
        # List checkpoint files
        checkpoint_files = sorted([
            os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir)
            if f.startswith('mountain_car_dceo_iter_') and f.endswith('.pt')
        ], key=lambda x: int(os.path.basename(x).split('_iter_')[1].split('.pt')[0]))
        
        # Add final checkpoint if it exists
        final_checkpoint = os.path.join(args.checkpoint_dir, 'mountain_car_dceo_final.pt')
        if os.path.exists(final_checkpoint):
            checkpoint_files.append(final_checkpoint)
        
        print(f"Found {len(checkpoint_files)} checkpoints to visualize")
        
        for checkpoint_path in checkpoint_files:
            visualize_checkpoint(checkpoint_path, args.save_dir)
    else:
        # Find the latest checkpoint
        checkpoint_files = sorted([
            os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir)
            if (f.startswith('mountain_car_dceo_iter_') or f == 'mountain_car_dceo_final.pt') and f.endswith('.pt')
        ], key=lambda x: os.path.getmtime(x), reverse=True)
        
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[0]
            print(f"Using latest checkpoint: {latest_checkpoint}")
            visualize_checkpoint(latest_checkpoint, args.save_dir)
        else:
            print("No checkpoint files found!")

if __name__ == '__main__':
    main()
