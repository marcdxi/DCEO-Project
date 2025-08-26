"""
Simple script to extract basic statistics from DCEO Mountain Car checkpoints.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

# Add import paths
import sys
sys.path.append('.')
sys.path.append('..')

from mountain_car_dceo.train_mountain_car_dceo import get_config, preprocess_state, handle_env_step, handle_env_reset
from pytorch_dceo_online import FullyOnlineDCEOAgent

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

def evaluate_agent(agent, env, num_episodes=5):
    """Evaluate agent performance."""
    rewards = []
    steps_list = []
    max_heights = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        max_height = -1.2  # Minimum position in Mountain Car
        done = False
        
        while not done:
            # Select action
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                processed_state = preprocess_state(state)
                processed_state = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                action = agent.act(processed_state, evaluation=True)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update metrics
            episode_reward += reward
            steps += 1
            max_height = max(max_height, next_state[0])
            
            # Check for success
            if next_state[0] >= 0.5:
                success_count += 1
                print(f"Episode {episode+1}: Success! Reached goal at step {steps}")
                done = True
            
            # Limit episode length
            if steps >= 500:
                done = True
                
            state = next_state
        
        rewards.append(episode_reward)
        steps_list.append(steps)
        max_heights.append(max_height)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}, Max Height = {max_height:.2f}")
    
    # Summary statistics
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)
    avg_max_height = np.mean(max_heights)
    success_rate = success_count / num_episodes
    
    return {
        'rewards': rewards,
        'steps': steps_list,
        'max_heights': max_heights,
        'mean_reward': avg_reward,
        'mean_steps': avg_steps,
        'mean_max_height': avg_max_height,
        'success_rate': success_rate
    }

def analyze_checkpoints(checkpoint_dir='./mountain_car_dceo/checkpoints', num_episodes=5):
    """Analyze all checkpoints in the directory."""
    # Create environment
    env = create_environment()
    
    # List checkpoint files
    checkpoint_files = sorted([
        os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
        if f.startswith('mountain_car_dceo_iter_') and f.endswith('.pt')
    ], key=lambda x: int(os.path.basename(x).split('_iter_')[1].split('.pt')[0]))
    
    # Add final checkpoint if it exists
    final_checkpoint = os.path.join(checkpoint_dir, 'mountain_car_dceo_final.pt')
    if os.path.exists(final_checkpoint):
        checkpoint_files.append(final_checkpoint)
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    all_results = []
    
    for checkpoint_path in checkpoint_files:
        print(f"\nEvaluating checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print(f"Trying to load with different approach...")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), pickle_module=pickle)
        
        # Get iteration number
        if 'iteration' in checkpoint:
            iteration = checkpoint['iteration']
        else:
            # Extract iteration number from filename
            iteration = int(os.path.basename(checkpoint_path).split('_iter_')[1].split('.pt')[0])
        
        # Create agent
        config = get_config()  # Use default config
        agent = create_agent(env, config)
        
        # Try to load state dict
        if 'agent_state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['agent_state_dict'])
        else:
            # Try different keys that might contain the state dict
            state_dict_keys = ['state_dict', 'model_state_dict', 'network_state_dict']
            for key in state_dict_keys:
                if key in checkpoint:
                    print(f"Loading state dict from '{key}' instead of 'agent_state_dict'")
                    agent.load_state_dict(checkpoint[key])
                    break
            else:
                print("WARNING: Could not find state dict in checkpoint, using random weights")
        
        # Evaluate agent
        results = evaluate_agent(agent, env, num_episodes=num_episodes)
        results['iteration'] = iteration
        all_results.append(results)
        
        print(f"\nIteration {iteration} Results:")
        print(f"  Average Reward: {results['mean_reward']:.2f}")
        print(f"  Average Steps: {results['mean_steps']:.2f}")
        print(f"  Average Max Height: {results['mean_max_height']:.2f}")
        print(f"  Success Rate: {results['success_rate']:.2f}")
    
    # Sort results by iteration
    all_results.sort(key=lambda x: x['iteration'])
    
    # Plot learning curves
    plot_learning_curves(all_results)
    
    # Print summary
    print("\nSummary of All Checkpoints:")
    for result in all_results:
        print(f"Iteration {result['iteration']}: " 
              f"Reward = {result['mean_reward']:.2f}, "
              f"Max Height = {result['mean_max_height']:.2f}, "
              f"Success = {result['success_rate']:.2f}")
    
    return all_results

def plot_learning_curves(all_results):
    """Plot learning curves for rewards and heights."""
    # Create results directory if it doesn't exist
    os.makedirs('./mountain_car_dceo/results', exist_ok=True)
    
    iterations = [r['iteration'] for r in all_results]
    mean_rewards = [r['mean_reward'] for r in all_results]
    mean_heights = [r['mean_max_height'] for r in all_results]
    success_rates = [r['success_rate'] for r in all_results]
    
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(iterations, mean_rewards, 'o-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Reward')
    plt.title('Reward Progress')
    plt.grid(True)
    
    # Plot max heights
    plt.subplot(3, 1, 2)
    plt.plot(iterations, mean_heights, 'o-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Goal Position')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Max Height')
    plt.title('Maximum Height Progress')
    plt.legend()
    plt.grid(True)
    
    # Plot success rates
    plt.subplot(3, 1, 3)
    plt.plot(iterations, success_rates, 'o-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Progress')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./mountain_car_dceo/results/learning_progress.png')
    print("Learning curves saved to ./mountain_car_dceo/results/learning_progress.png")
    plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract statistics from DCEO Mountain Car checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./mountain_car_dceo/checkpoints',
                        help='Directory containing checkpoint files')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to evaluate each checkpoint')
    
    args = parser.parse_args()
    
    all_results = analyze_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        num_episodes=args.num_episodes
    )
