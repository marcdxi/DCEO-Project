"""
Analysis script for DCEO Mountain Car experiments.

This script loads checkpoint files from training and generates various analyses:
1. Learning curves (reward vs. time)
2. State coverage statistics 
3. Option utilization statistics
4. Eigenvector visualization and comparison
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

# Add import paths
import sys
sys.path.append('.')
sys.path.append('..')

# Import from the training script
from mountain_car_dceo.train_mountain_car_dceo import preprocess_state, handle_env_step, handle_env_reset, get_config
from pytorch_dceo_online import FullyOnlineDCEOAgent

# Helper functions for environment and agent creation
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

def create_agent(env, config):
    """Create a DCEO agent with the given configuration."""
    # Get input shape from preprocessed state
    sample_state = handle_env_reset(env)
    preprocessed_shape = preprocess_state(sample_state).shape
    
    # Create agent
    agent = FullyOnlineDCEOAgent(
        input_shape=preprocessed_shape,
        num_actions=env.action_space.n,
        **config
    )
    
    return agent

# Create state discretization for eigenvector analysis
def discretize_state(state, env_min=np.array([-1.2, -0.07]), env_max=np.array([0.6, 0.07]), bins=(20, 20)):
    """Discretize state for creating transition matrix."""
    state_adj = (state - env_min) / (env_max - env_min)
    state_adj = np.clip(state_adj, 0, 1)
    discretized = tuple(min(int(s * b), b - 1) for s, b in zip(state_adj, bins))
    return discretized

def state_to_index(state, bins=(20, 20)):
    """Convert discretized state to single index."""
    return state[0] * bins[1] + state[1]

def compute_true_eigenvectors(env, num_options=5, episodes=300, max_steps=200, bins=(20, 20)):
    """Compute the true eigenvectors of the environment's transition matrix."""
    print("Computing true eigenvectors of the environment...")
    
    # Initialize transition counts
    total_states = bins[0] * bins[1]
    transition_counts = np.zeros((total_states, total_states))
    
    for _ in tqdm(range(episodes)):
        state, _ = env.reset()
        discretized_state = discretize_state(state, bins=bins)
        state_idx = state_to_index(discretized_state, bins)
        
        for _ in range(max_steps):
            action = env.action_space.sample()  # Random policy
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_discretized = discretize_state(next_state, bins=bins)
            next_idx = state_to_index(next_discretized, bins)
            
            # Update transition count
            transition_counts[state_idx, next_idx] += 1
            
            state = next_state
            discretized_state = next_discretized
            state_idx = next_idx
            
            if done:
                break
    
    # Convert counts to probabilities (row stochastic)
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for states never visited
    row_sums[row_sums == 0] = 1.0
    transition_matrix = transition_counts / row_sums
    
    # Compute the largest eigenvectors of the transition matrix
    sparse_matrix = csr_matrix(transition_matrix)
    _, eigenvectors = eigs(sparse_matrix.T, k=num_options+1, which='LM')
    
    # Discard the first eigenvector (constant)
    eigenvectors = np.real(eigenvectors[:, 1:num_options+1])
    
    # Normalize
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    
    return eigenvectors

def extract_agent_eigenvectors(agent, bins=(20, 20)):
    """Extract the eigenvectors learned by the agent's representation network."""
    print("Extracting agent's learned eigenvectors...")
    
    # Create a grid of states covering the state space
    pos_range = np.linspace(-1.2, 0.6, bins[0])
    vel_range = np.linspace(-0.07, 0.07, bins[1])
    
    state_grid = np.zeros((bins[0] * bins[1], 2))
    idx = 0
    for pos in pos_range:
        for vel in vel_range:
            state_grid[idx] = [pos, vel]
            idx += 1
    
    # Get agent's representations for these states
    agent_eigenvectors = np.zeros((bins[0] * bins[1], agent.num_options))
    
    # Process in batches to avoid memory issues
    batch_size = 128
    num_batches = (state_grid.shape[0] + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, state_grid.shape[0])
        batch_states = state_grid[start_idx:end_idx]
        
        # Convert to tensor
        state_tensor = torch.tensor(batch_states, dtype=torch.float32).to(agent.device)
        
        # Get representation (this depends on how your agent processes states)
        with torch.no_grad():
            # Preprocess states one at a time to match training
            processed_states = torch.stack([preprocess_state(s) for s in batch_states])
            processed_states = processed_states.to(agent.device)
            
            # Extract representations from the agent
            representations = agent.rainbow_net.encoder_net(processed_states).detach()
        
        agent_eigenvectors[start_idx:end_idx] = representations.cpu().numpy()
    
    return agent_eigenvectors, state_grid

def compute_eigenvector_similarity(true_eigenvectors, agent_eigenvectors):
    """Compute similarity between true and learned eigenvectors."""
    similarity_matrix = np.zeros((true_eigenvectors.shape[1], agent_eigenvectors.shape[1]))
    
    for i in range(true_eigenvectors.shape[1]):
        for j in range(agent_eigenvectors.shape[1]):
            # Compute absolute cosine similarity (eigenspaces can have flipped signs)
            similarity = np.abs(np.dot(true_eigenvectors[:, i], agent_eigenvectors[:, j]))
            similarity /= (np.linalg.norm(true_eigenvectors[:, i]) * np.linalg.norm(agent_eigenvectors[:, j]))
            similarity_matrix[i, j] = similarity
    
    return similarity_matrix

def evaluate_checkpoint(checkpoint_path, env, num_episodes=5, render=False):
    """Evaluate a checkpoint and return various metrics."""
    print(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Trying to load with different approach...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), pickle_module=pickle)
    
    # Set default iteration if not in checkpoint
    if 'iteration' not in checkpoint:
        iter_num = int(os.path.basename(checkpoint_path).split('_iter_')[1].split('.pt')[0]) 
        checkpoint['iteration'] = iter_num
    
    # Create agent with same parameters
    agent = create_agent(env, checkpoint['config'])
    agent.load_state_dict(checkpoint['agent_state_dict'])
    
    # Evaluation metrics
    total_rewards = []
    episode_lengths = []
    max_heights = []
    visited_states = set()
    option_usage = defaultdict(int)
    option_durations = defaultdict(list)
    transitions = []  # Store (state, action, next_state) for building graph
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        max_height = -1.2  # Minimum position
        
        current_option = None
        option_step_count = 0
        
        done = False
        while not done:
            # Track visited states
            discretized = discretize_state(state)
            visited_states.add(discretized)
            
            # Select action using agent
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                processed_state = agent.preprocess_states(state_tensor)
                action, option_info = agent.select_action(processed_state, evaluation=True)
            
            # Track option usage
            new_option = option_info.get('option_id', None)
            if new_option != current_option:
                if current_option is not None:
                    option_durations[current_option].append(option_step_count)
                current_option = new_option
                option_step_count = 1
                if current_option is not None:
                    option_usage[current_option] += 1
            else:
                option_step_count += 1
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition for graph building
            transitions.append((state, action, next_state))
            
            # Update metrics
            episode_reward += reward
            steps += 1
            max_height = max(max_height, next_state[0])
            
            # Render if requested
            if render:
                env.render()
            
            state = next_state
            
            if done:
                if current_option is not None:
                    option_durations[current_option].append(option_step_count)
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        max_heights.append(max_height)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}, Max Height = {max_height:.2f}")
    
    # Calculate average option durations
    avg_option_durations = {opt: np.mean(durations) for opt, durations in option_durations.items()}
    
    # Calculate state coverage percentage
    total_possible_states = 20 * 20  # Assuming 20x20 discretization
    coverage_percentage = (len(visited_states) / total_possible_states) * 100
    
    # Get learned eigenvectors from the agent
    learned_eigenvectors, state_grid = extract_agent_eigenvectors(agent)
    
    results = {
        'iteration': checkpoint.get('iteration', 0),
        'rewards': total_rewards,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'steps': episode_lengths,
        'mean_steps': np.mean(episode_lengths),
        'max_heights': max_heights,
        'mean_max_height': np.mean(max_heights),
        'state_coverage': coverage_percentage,
        'option_usage': dict(option_usage),
        'option_durations': avg_option_durations,
        'transitions': transitions,
        'learned_eigenvectors': learned_eigenvectors,
        'state_grid': state_grid
    }
    
    return results

def plot_learning_curve(all_results):
    """Plot learning curve showing reward vs iteration."""
    iterations = [r['iteration'] for r in all_results]
    mean_rewards = [r['mean_reward'] for r in all_results]
    std_rewards = [r['std_reward'] for r in all_results]
    mean_heights = [r['mean_max_height'] for r in all_results]
    
    plt.figure(figsize=(12, 6))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.errorbar(iterations, mean_rewards, yerr=std_rewards, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Reward')
    plt.title('Learning Curve - Rewards')
    plt.grid(True)
    
    # Plot max heights
    plt.subplot(1, 2, 2)
    plt.plot(iterations, mean_heights, marker='o', linestyle='-')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Goal Position')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Max Height')
    plt.title('Learning Curve - Max Heights')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./mountain_car_dceo/results/plots/learning_curve.png')
    plt.close()

def plot_state_coverage(all_results):
    """Plot state coverage over iterations."""
    iterations = [r['iteration'] for r in all_results]
    coverage = [r['state_coverage'] for r in all_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, coverage, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('State Coverage (%)')
    plt.title('State Space Coverage')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./mountain_car_dceo/results/plots/state_coverage.png')
    plt.close()

def plot_option_utilization(all_results):
    """Plot option utilization over iterations."""
    iterations = [r['iteration'] for r in all_results]
    num_options = len(all_results[0]['option_usage'])
    
    # Get option usage counts across iterations
    option_counts = []
    option_durations = []
    
    for r in all_results:
        counts = [r['option_usage'].get(i, 0) for i in range(num_options)]
        option_counts.append(counts)
        
        durations = [r['option_durations'].get(i, 0) for i in range(num_options)]
        option_durations.append(durations)
    
    option_counts = np.array(option_counts)
    option_durations = np.array(option_durations)
    
    # Plot option usage counts
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    for i in range(num_options):
        plt.plot(iterations, option_counts[:, i], marker='o', linestyle='-', label=f'Option {i}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Usage Count')
    plt.title('Option Usage Frequency')
    plt.legend()
    plt.grid(True)
    
    # Plot option durations
    plt.subplot(2, 1, 2)
    for i in range(num_options):
        plt.plot(iterations, option_durations[:, i], marker='o', linestyle='-', label=f'Option {i}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Duration (steps)')
    plt.title('Option Duration')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./mountain_car_dceo/results/plots/option_utilization.png')
    plt.close()

def plot_eigenvector_comparison(all_results, true_eigenvectors):
    """Plot comparison between true and learned eigenvectors."""
    iterations = [r['iteration'] for r in all_results]
    num_options = all_results[0]['learned_eigenvectors'].shape[1]
    
    # Calculate similarity at each iteration
    similarities = []
    best_matches = []
    
    for r in all_results:
        similarity = compute_eigenvector_similarity(true_eigenvectors, r['learned_eigenvectors'])
        similarities.append(similarity)
        
        # For each true eigenvector, find the best matching learned one
        best_match_indices = np.argmax(similarity, axis=1)
        best_match_values = np.max(similarity, axis=1)
        best_matches.append(best_match_values)
    
    best_matches = np.array(best_matches)
    
    # Plot similarity trends
    plt.figure(figsize=(12, 6))
    
    for i in range(true_eigenvectors.shape[1]):
        plt.plot(iterations, best_matches[:, i], marker='o', linestyle='-', 
                 label=f'True Eigenvector {i+1}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Between True and Learned Eigenvectors')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('./mountain_car_dceo/results/plots/eigenvector_similarity.png')
    plt.close()
    
    # Visualize eigenvectors at final iteration
    final_result = all_results[-1]
    state_grid = final_result['state_grid']
    learned_eigenvectors = final_result['learned_eigenvectors']
    
    # Reshape for visualization
    reshaped_true = []
    reshaped_learned = []
    
    # Reshape eigenvectors back to 2D grid for visualization
    for i in range(min(5, num_options)):
        true_vec = true_eigenvectors[:, i].reshape(20, 20)
        
        # Find best matching learned eigenvector
        best_idx = np.argmax(similarities[-1][i])
        learned_vec = learned_eigenvectors[:, best_idx].reshape(20, 20)
        
        reshaped_true.append(true_vec)
        reshaped_learned.append(learned_vec)
    
    # Plot true vs learned eigenvectors
    plt.figure(figsize=(15, 12))
    
    for i in range(len(reshaped_true)):
        # True eigenvector
        plt.subplot(2, len(reshaped_true), i + 1)
        plt.imshow(reshaped_true[i], cmap='coolwarm')
        plt.title(f'True Eigenvector {i+1}')
        plt.colorbar()
        
        # Learned eigenvector
        plt.subplot(2, len(reshaped_true), i + 1 + len(reshaped_true))
        plt.imshow(reshaped_learned[i], cmap='coolwarm')
        plt.title(f'Learned Eigenvector (sim={similarities[-1][i, np.argmax(similarities[-1][i])]:0.2f})')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('./mountain_car_dceo/results/visualizations/eigenvector_visualization.png')
    plt.close()

def visualize_state_action_values(checkpoint_path, env, bins=(20, 20)):
    """Visualize the state-action values learned by the agent."""
    print(f"Visualizing state-action values for: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Create agent with same parameters
    agent = create_agent(env, checkpoint['config'])
    agent.load_state_dict(checkpoint['agent_state_dict'])
    
    # Create a grid of states covering the state space
    pos_range = np.linspace(-1.2, 0.6, bins[0])
    vel_range = np.linspace(-0.07, 0.07, bins[1])
    
    # Arrays to store Q-values for each action
    q_values = np.zeros((3, bins[0], bins[1]))
    
    # Get Q-values for each state
    for i, pos in enumerate(pos_range):
        for j, vel in enumerate(vel_range):
            state = np.array([pos, vel])
            
            # Get Q-values from agent
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                processed_state = agent.preprocess_states(state_tensor)
                q_vals = agent.get_q_values(processed_state).cpu().numpy()[0]
            
            # Store Q-values
            for a in range(3):
                q_values[a, i, j] = q_vals[a]
    
    # Create visualization
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot Q-values for each action
    action_names = ['Left', 'No-op', 'Right']
    for a in range(3):
        im = axs[a].imshow(q_values[a], origin='lower', cmap='viridis',
                         extent=[-1.2, 0.6, -0.07, 0.07])
        axs[a].set_title(f'Q-values for Action: {action_names[a]}')
        axs[a].set_xlabel('Position')
        axs[a].set_ylabel('Velocity')
        plt.colorbar(im, ax=axs[a])
    
    # Plot max Q-value action (policy)
    best_actions = np.argmax(q_values, axis=0)
    im = axs[3].imshow(best_actions, origin='lower', cmap='plasma', 
                     extent=[-1.2, 0.6, -0.07, 0.07], vmin=0, vmax=2)
    axs[3].set_title('Best Action (0=Left, 1=No-op, 2=Right)')
    axs[3].set_xlabel('Position')
    axs[3].set_ylabel('Velocity')
    cbar = plt.colorbar(im, ax=axs[3], ticks=[0, 1, 2])
    cbar.set_ticklabels(['Left', 'No-op', 'Right'])
    
    plt.tight_layout()
    plt.savefig('./mountain_car_dceo/results/visualizations/q_values.png')
    plt.close()

def visualize_option_policies(checkpoint_path, env, bins=(20, 20)):
    """Visualize the policies of each option."""
    print(f"Visualizing option policies for: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Create agent with same parameters
    agent = create_agent(env, checkpoint['config'])
    agent.load_state_dict(checkpoint['agent_state_dict'])
    
    num_options = agent.num_options
    
    # Create a grid of states covering the state space
    pos_range = np.linspace(-1.2, 0.6, bins[0])
    vel_range = np.linspace(-0.07, 0.07, bins[1])
    
    # Arrays to store option policies and termination probabilities
    option_policies = np.zeros((num_options, bins[0], bins[1]))
    termination_probs = np.zeros((num_options, bins[0], bins[1]))
    
    # Get policies for each state
    for i, pos in enumerate(pos_range):
        for j, vel in enumerate(vel_range):
            state = np.array([pos, vel])
            
            # Get option policies from agent
            with torch.no_grad():
                # Preprocess state
                processed_state = preprocess_state(state).unsqueeze(0).to(agent.device)
                
                # For each option, get the option's policy and termination probability
                for opt in range(num_options):
                    # Use the agent's option networks to get action values for this option
                    q_values = agent.rainbow_net.option_value_networks[opt](processed_state).mean(dim=2).sum(dim=0)
                    action = torch.argmax(q_values).item()
                    option_policies[opt, i, j] = action
                    
                    # Get termination probability
                    term_logits = agent.rainbow_net.termination_networks[opt](processed_state)
                    term_prob = torch.sigmoid(term_logits)
                    termination_probs[opt, i, j] = term_prob.cpu().numpy()[0]
    
    # Create visualization
    rows = 2  # First row for policies, second for termination probs
    cols = num_options
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    action_names = ['Left', 'No-op', 'Right']
    for opt in range(num_options):
        # Plot policy
        im = axs[0, opt].imshow(option_policies[opt], origin='lower', cmap='plasma',
                             extent=[-1.2, 0.6, -0.07, 0.07], vmin=0, vmax=2)
        axs[0, opt].set_title(f'Option {opt} Policy')
        axs[0, opt].set_xlabel('Position')
        axs[0, opt].set_ylabel('Velocity')
        cbar = plt.colorbar(im, ax=axs[0, opt], ticks=[0, 1, 2])
        cbar.set_ticklabels(action_names)
        
        # Plot termination probability
        im = axs[1, opt].imshow(termination_probs[opt], origin='lower', cmap='viridis',
                             extent=[-1.2, 0.6, -0.07, 0.07], vmin=0, vmax=1)
        axs[1, opt].set_title(f'Option {opt} Termination Probability')
        axs[1, opt].set_xlabel('Position')
        axs[1, opt].set_ylabel('Velocity')
        plt.colorbar(im, ax=axs[1, opt])
    
    plt.tight_layout()
    plt.savefig('./mountain_car_dceo/results/visualizations/option_policies.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze DCEO Mountain Car results')
    parser.add_argument('--checkpoint_dir', type=str, default='./mountain_car_dceo/checkpoints',
                        help='Directory containing checkpoint files')
    parser.add_argument('--output_dir', type=str, default='./mountain_car_dceo/results',
                        help='Directory to save analysis results')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Number of episodes to evaluate each checkpoint')
    parser.add_argument('--render', action='store_true', help='Render evaluation episodes')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create required subdirectories
    for subdir in ['plots', 'data', 'visualizations']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # List checkpoint files
    checkpoint_files = sorted([
        os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir)
        if f.startswith('mountain_car_dceo_iter_') and f.endswith('.pt')
    ], key=lambda x: int(os.path.basename(x).split('_iter_')[1].split('.pt')[0]))
    
    # Add final checkpoint if it exists
    final_checkpoint = os.path.join(args.checkpoint_dir, 'mountain_car_dceo_final.pt')
    if os.path.exists(final_checkpoint):
        checkpoint_files.append(final_checkpoint)
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Create environment
    env = create_environment()
    
    # Compute true eigenvectors
    try:
        print("Attempting to compute true eigenvectors (this may take some time)...")
        true_eigenvectors = compute_true_eigenvectors(env)
    except Exception as e:
        print(f"Error computing true eigenvectors: {e}")
        print("Continuing analysis without true eigenvectors.")
        true_eigenvectors = None
    
    # Evaluate each checkpoint
    all_results = []
    for checkpoint_path in checkpoint_files:
        results = evaluate_checkpoint(checkpoint_path, env, num_episodes=args.eval_episodes, render=args.render)
        all_results.append(results)
        
        # Save results
        iteration = results['iteration']
        with open(f'{args.output_dir}/data/results_iter_{iteration}.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    # Generate plots
    print("Generating learning curve plot...")
    plot_learning_curve(all_results)
    
    print("Generating state coverage plot...")
    plot_state_coverage(all_results)
    
    print("Generating option utilization plot...")
    plot_option_utilization(all_results)
    
    # Only generate eigenvector comparison if we have true eigenvectors
    if true_eigenvectors is not None:
        print("Generating eigenvector comparison plot...")
        plot_eigenvector_comparison(all_results, true_eigenvectors)
    else:
        print("Skipping eigenvector comparison (no true eigenvectors)")

    
    # Visualize state-action values and option policies for final checkpoint
    print("Visualizing Q-values and option policies...")
    if checkpoint_files:
        final_checkpoint = checkpoint_files[-1]
        visualize_state_action_values(final_checkpoint, env)
        visualize_option_policies(final_checkpoint, env)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
