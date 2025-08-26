"""
Script to analyze and visualize the options learned by the fully online DCEO agent
on the Mountain Car environment.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import gym
import seaborn as sns

from pytorch_dceo_online import FullyOnlineDCEOAgent
from train_mountain_car_dceo import preprocess_state, handle_env_reset, handle_env_step

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def visualize_option_policies(agent, env, resolution=50):
    """Visualize the learned option policies across the state space.
    
    Args:
        agent: Trained FullyOnlineDCEOAgent
        env: Mountain Car environment
        resolution: Resolution of the visualization grid
    """
    # Create a grid of the state space
    positions = np.linspace(-1.2, 0.6, resolution)
    velocities = np.linspace(-0.07, 0.07, resolution)
    
    # Initialize grid for each option
    option_grids = [np.zeros((resolution, resolution, env.action_space.n)) 
                   for _ in range(agent.num_options)]
    
    # Compute option policies for each state
    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            # Create state
            state = np.array([pos, vel])
            
            # Preprocess state
            processed_state = preprocess_state(state)
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(device)
            
            # For each option, get the action probabilities
            for opt_idx in range(agent.num_options):
                option_net = agent.option_nets[opt_idx]
                if agent.distributional:
                    logits = option_net(state_tensor)
                    probs = torch.sum(F.softmax(logits, dim=-1) * agent.supports.view(1, 1, -1), dim=-1)
                else:
                    probs = option_net(state_tensor)
                
                option_grids[opt_idx][i, j] = probs.detach().cpu().numpy()[0]
    
    # Visualize each option's preferred action
    fig, axes = plt.subplots(1, agent.num_options, figsize=(4*agent.num_options, 4))
    if agent.num_options == 1:
        axes = [axes]
    
    action_names = ["Left", "No-op", "Right"]
    
    for opt_idx, ax in enumerate(axes):
        # Get preferred action for each state
        preferred_actions = np.argmax(option_grids[opt_idx], axis=2)
        
        # Plot
        im = ax.imshow(preferred_actions, origin='lower', aspect='auto', 
                      extent=[-1.2, 0.6, -0.07, 0.07])
        ax.set_title(f"Option {opt_idx} Policy")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        
        # Add colorbar with action labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.set_ticklabels(action_names)
    
    plt.suptitle("Option Policies Across State Space", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'option_policies.png'))
    plt.show()


def visualize_option_values(agent, env, resolution=50):
    """Visualize the learned option values across the state space.
    
    Args:
        agent: Trained FullyOnlineDCEOAgent
        env: Mountain Car environment
        resolution: Resolution of the visualization grid
    """
    # Create a grid of the state space
    positions = np.linspace(-1.2, 0.6, resolution)
    velocities = np.linspace(-0.07, 0.07, resolution)
    
    # Initialize grid for each option
    option_value_grids = [np.zeros((resolution, resolution)) 
                         for _ in range(agent.num_options)]
    main_policy_grid = np.zeros((resolution, resolution))
    
    # Compute option values for each state
    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            # Create state
            state = np.array([pos, vel])
            
            # Preprocess state
            processed_state = preprocess_state(state)
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(device)
            
            # Main policy value
            if agent.distributional:
                logits = agent.policy_net(state_tensor)
                probs = torch.sum(F.softmax(logits, dim=-1) * agent.supports.view(1, 1, -1), dim=-1)
                main_policy_grid[i, j] = torch.max(probs).item()
            else:
                q_values = agent.policy_net(state_tensor)
                main_policy_grid[i, j] = torch.max(q_values).item()
            
            # Option values
            for opt_idx in range(agent.num_options):
                option_net = agent.option_nets[opt_idx]
                if agent.distributional:
                    logits = option_net(state_tensor)
                    probs = torch.sum(F.softmax(logits, dim=-1) * agent.supports.view(1, 1, -1), dim=-1)
                    option_value_grids[opt_idx][i, j] = torch.max(probs).item()
                else:
                    q_values = option_net(state_tensor)
                    option_value_grids[opt_idx][i, j] = torch.max(q_values).item()
    
    # Visualize values for each option
    fig, axes = plt.subplots(1, agent.num_options + 1, figsize=(4*(agent.num_options + 1), 4))
    
    # Plot main policy values
    im = axes[0].imshow(main_policy_grid, origin='lower', aspect='auto', 
                      extent=[-1.2, 0.6, -0.07, 0.07], cmap='viridis')
    axes[0].set_title("Main Policy Values")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Velocity")
    plt.colorbar(im, ax=axes[0])
    
    # Plot option values
    for opt_idx in range(agent.num_options):
        im = axes[opt_idx + 1].imshow(option_value_grids[opt_idx], origin='lower', aspect='auto', 
                                    extent=[-1.2, 0.6, -0.07, 0.07], cmap='viridis')
        axes[opt_idx + 1].set_title(f"Option {opt_idx} Values")
        axes[opt_idx + 1].set_xlabel("Position")
        if opt_idx > 0:  # Only show y-label on first plot
            axes[opt_idx + 1].set_ylabel("")
        plt.colorbar(im, ax=axes[opt_idx + 1])
    
    plt.suptitle("Value Functions Across State Space", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'option_values.png'))
    plt.show()


def visualize_option_usage(agent, env, num_episodes=10):
    """Visualize which option is used in which part of the state space.
    
    Args:
        agent: Trained FullyOnlineDCEOAgent
        env: Mountain Car environment
        num_episodes: Number of episodes to collect data
    """
    # Collect state-option data
    states = []
    options = []
    
    for episode in range(num_episodes):
        state = handle_env_reset(env)
        done = False
        
        while not done:
            # Preprocess state
            processed_state = preprocess_state(state)
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(device)
            
            # Record raw state
            states.append(state.copy())
            
            # Select action and get the current option
            action = agent.select_action(state_tensor, eval_mode=True)
            options.append(agent.cur_opt)
            
            # Step environment
            next_state, reward, done, _ = handle_env_step(env, action)
            state = next_state
            
            if len(states) > 10000:  # Limit data collection
                break
    
    # Convert to numpy arrays
    states = np.array(states)
    options = np.array(options)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot points colored by option
    for opt in range(agent.num_options):
        mask = options == opt
        if np.any(mask):
            plt.scatter(states[mask, 0], states[mask, 1], label=f'Option {opt}', alpha=0.5)
    
    # Add goal line
    plt.axvline(x=0.5, color='r', linestyle='--', label='Goal threshold')
    
    plt.title('Option Usage Across State Space')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'option_usage.png'))
    plt.show()


def visualize_representations(agent, env, resolution=20):
    """Visualize the learned representations across the state space.
    
    Args:
        agent: Trained FullyOnlineDCEOAgent
        env: Mountain Car environment
        resolution: Resolution of the visualization grid
    """
    # Create a grid of the state space
    positions = np.linspace(-1.2, 0.6, resolution)
    velocities = np.linspace(-0.07, 0.07, resolution)
    
    # Compute the representations for each state in the grid
    rep_vectors = np.zeros((resolution, resolution, agent.rep_dim))
    
    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            # Create state
            state = np.array([pos, vel])
            
            # Preprocess state
            processed_state = preprocess_state(state)
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(device)
            
            # Compute representation
            with torch.no_grad():
                rep = agent.rep_net(state_tensor)
            
            rep_vectors[i, j] = rep.cpu().numpy()[0]
    
    # Visualize first few dimensions of the representation
    num_dims_to_plot = min(6, agent.rep_dim)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for dim in range(num_dims_to_plot):
        im = axes[dim].imshow(rep_vectors[:, :, dim], origin='lower', aspect='auto', 
                           extent=[-1.2, 0.6, -0.07, 0.07], cmap='coolwarm')
        axes[dim].set_title(f'Representation Dimension {dim}')
        axes[dim].set_xlabel('Position')
        axes[dim].set_ylabel('Velocity')
        plt.colorbar(im, ax=axes[dim])
    
    plt.suptitle('Learned Representation Dimensions', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'representation_dimensions.png'))
    plt.show()
    
    # Perform dimensionality reduction for visualization
    from sklearn.manifold import TSNE
    
    # Flatten the grid for TSNE
    flat_states = [(pos, vel) for pos in positions for vel in velocities]
    flat_reps = rep_vectors.reshape(-1, agent.rep_dim)
    
    # Apply TSNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_reps = tsne.fit_transform(flat_reps)
    
    # Plot reduced representations
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(reduced_reps[:, 0], reduced_reps[:, 1], 
                        c=[p[0] for p in flat_states],  # Color by position
                        cmap='viridis')
    plt.colorbar(scatter, label='Position')
    plt.title('t-SNE Visualization of Learned Representations')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'tsne_representations.png'))
    plt.show()


def analyze_trajectories(agent, env, num_episodes=5):
    """Analyze and visualize agent trajectories with option usage.
    
    Args:
        agent: Trained FullyOnlineDCEOAgent
        env: Mountain Car environment
        num_episodes: Number of episodes to analyze
    """
    # Set up figure for visualizing trajectories
    plt.figure(figsize=(15, 10))
    
    for episode in range(num_episodes):
        state = handle_env_reset(env)
        done = False
        
        # Storage for trajectory
        positions = []
        velocities = []
        option_used = []
        
        steps = 0
        success = False
        
        while not done and steps < 1000:
            # Preprocess state
            processed_state = preprocess_state(state)
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(device)
            
            # Select action and record option
            action = agent.select_action(state_tensor, eval_mode=True)
            current_option = agent.cur_opt
            
            # Store state and option
            positions.append(state[0])
            velocities.append(state[1])
            option_used.append(current_option)
            
            # Step environment
            next_state, reward, done, _ = handle_env_step(env, action)
            state = next_state
            
            # Check for success
            if state[0] >= 0.5:
                success = True
                done = True
                
            steps += 1
        
        # Plot trajectory
        plt.subplot(num_episodes, 1, episode + 1)
        
        # Segment trajectory by option
        segments = []
        current_segment = []
        current_option = option_used[0]
        
        for i in range(len(positions)):
            if option_used[i] == current_option:
                current_segment.append((positions[i], velocities[i]))
            else:
                segments.append((current_segment, current_option))
                current_segment = [(positions[i], velocities[i])]
                current_option = option_used[i]
        
        # Add last segment
        if current_segment:
            segments.append((current_segment, current_option))
        
        # Plot each segment with a different color for each option
        for segment, opt in segments:
            seg_pos, seg_vel = zip(*segment)
            plt.plot(seg_pos, seg_vel, '.-', linewidth=2, label=f'Option {opt}' if opt is not None else 'No Option')
        
        # Add goal line
        plt.axvline(x=0.5, color='r', linestyle='--')
        
        # Add markers for start and end
        plt.plot(positions[0], velocities[0], 'go', markersize=10, label='Start')
        plt.plot(positions[-1], velocities[-1], 'ro', markersize=10, label='End')
        
        # Add title and labels
        outcome = "Success" if success else "Failure"
        plt.title(f'Episode {episode + 1}: {outcome} ({steps} steps)')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.grid(True)
        
        # Only show legend on first plot to avoid clutter
        if episode == 0:
            plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'agent_trajectories.png'))
    plt.show()


def load_agent(checkpoint_path, input_shape, num_actions):
    """Load a trained agent from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        input_shape: Shape of the preprocessed state input
        num_actions: Number of actions in the environment
        
    Returns:
        agent: Loaded FullyOnlineDCEOAgent
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration from the checkpoint
    config = {}
    
    # Determine number of options from the checkpoint
    num_options = len(checkpoint['option_state_dicts'])
    
    # Create a default config
    config = {
        'num_options': num_options,
        'rep_dim': 20,  # Default
        'noisy': True,
        'dueling': True,
        'distributional': True,
        'num_atoms': 51,
        'v_min': -10,
        'v_max': 10,
    }
    
    # Create agent
    agent = FullyOnlineDCEOAgent(
        input_shape=input_shape,
        num_actions=num_actions,
        **config
    )
    
    # Load state dictionaries
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.rep_net.load_state_dict(checkpoint['rep_state_dict'])
    
    for i, state_dict in enumerate(checkpoint['option_state_dicts']):
        agent.option_nets[i].load_state_dict(state_dict)
    
    # Copy weights to target networks
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    for i in range(num_options):
        agent.option_targets[i].load_state_dict(agent.option_nets[i].state_dict())
    
    print(f"Loaded agent with {num_options} options from {checkpoint_path}")
    return agent


def main():
    """Main function to run option analysis."""
    parser = argparse.ArgumentParser(description='Analyze options learned by DCEO agent on Mountain Car')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    
    args = parser.parse_args()
    
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
    
    # Get input shape from preprocessed state
    sample_state = handle_env_reset(env)
    preprocessed_shape = preprocess_state(sample_state).shape
    print(f"Preprocessed state shape: {preprocessed_shape}")
    
    # Load agent
    agent = load_agent(args.checkpoint, preprocessed_shape, env.action_space.n)
    
    # Set agent to evaluation mode
    agent.eval_mode = True
    
    # Run analysis functions
    print("\nAnalyzing option policies...")
    visualize_option_policies(agent, env)
    
    print("\nAnalyzing option values...")
    visualize_option_values(agent, env)
    
    print("\nAnalyzing option usage...")
    visualize_option_usage(agent, env)
    
    print("\nAnalyzing learned representations...")
    visualize_representations(agent, env)
    
    print("\nAnalyzing agent trajectories...")
    analyze_trajectories(agent, env)
    
    env.close()
    print("\nAnalysis complete!")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
