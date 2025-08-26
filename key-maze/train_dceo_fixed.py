# Import necessary libraries
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, clear_output

# Add parent directory to path if needed
if os.path.exists('../'):
    sys.path.append('../')

# Import your modules - Using KeyMazeWrapper instead of DCEOKeyMazeWrapper
from key_door_maze_env import KeyDoorMazeEnv
from key_maze_wrapper import KeyMazeWrapper  # Changed from DCEOKeyMazeWrapper to KeyMazeWrapper
from key_maze_dceo_agent import KeyMazeDCEOAgent

# Set random seed for reproducibility
def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
set_seeds(42)

# Configuration
maze_size = 10
num_keys = 1
max_steps = 300
episodes = 2000
eval_interval = 20
eval_episodes = 10
fixed_maze = True
maze_seed = 42

print(f"Training for {episodes} episodes")
print(f"Evaluating every {eval_interval} episodes with {eval_episodes} eval episodes")
print(f"Maze size: {maze_size}, Keys: {num_keys}, Max steps: {max_steps}")

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create environment
env = KeyDoorMazeEnv(
    maze_size=maze_size, 
    num_keys=num_keys, 
    max_steps=max_steps,
    use_fixed_layout=fixed_maze,
    use_fixed_seed=fixed_maze,
    fixed_seed=maze_seed
)

# Wrap environment
env = KeyMazeWrapper(  # Changed from DCEOKeyMazeWrapper to KeyMazeWrapper
    env,
    frame_stack=4,
    resize_shape=(84, 84),
    proximity_reward=True
)

if fixed_maze:
    print(f"Using fixed maze layout with seed {maze_seed}")

# Get observation shape
obs, _ = env.reset()
pytorch_shape = obs.shape
print(f"Environment: Key-Door Maze")
print(f"Observation shape: {obs.shape}, PyTorch shape: {pytorch_shape}")
print(f"Action size: {env.action_space.n}")

# Create DCEO agent
print("\n===== Creating Rainbow DCEO Agent =====")
# Match hyperparameters from the DCEO paper and JAX implementation
num_options = 8       # Paper uses 8 eigenoptions
rep_dim = 8           # Representation dimension should match number of options
gamma = 0.99          # Discount factor from the paper
learning_rate = 1e-4  # Learning rate from the paper (0.0001)
buffer_size = 100000  # Larger buffer size as in the paper

dceo_agent = KeyMazeDCEOAgent(
    state_shape=pytorch_shape,
    action_size=env.action_space.n,
    buffer_size=buffer_size,    
    learning_rate=learning_rate, 
    gamma=gamma,               
    num_options=num_options,   
    rep_dim=rep_dim,          
    option_prob=0.8,           
    option_duration=15,        
    prioritized_replay=True,   
    noisy_nets=True,           
    dueling=True,              
    double_dqn=True,           
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995
)

# Move agent to device
dceo_agent.to(device)

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate an agent on the environment without exploration."""
    total_reward = 0
    success_count = 0
    key_count = 0
    door_count = 0
    coverage_total = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select action without exploration
            action = agent.select_action(obs, epsilon=0.001, eval_mode=True)
            
            # Take action in environment - handle both gym interface versions
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
                
            episode_reward += reward
        
        # Check if goal was reached
        if info.get('goal_reached', False):
            success_count += 1
            
        # Count keys collected and doors opened
        key_count += info.get('keys_collected', 0)
        door_count += info.get('doors_opened', 0)
        
        # Track coverage
        coverage_stats = env.get_coverage_stats()
        coverage_total += coverage_stats['coverage']
        
        total_reward += episode_reward
    
    # Calculate average metrics
    avg_reward = total_reward / num_episodes
    success_rate = success_count / num_episodes
    avg_keys = key_count / num_episodes
    avg_doors = door_count / num_episodes
    avg_coverage = coverage_total / num_episodes
    
    return {
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'avg_keys': avg_keys,
        'avg_doors': avg_doors,
        'avg_coverage': avg_coverage
    }

def visualize_and_train():
    # Initialize metrics tracking
    rewards = []
    eval_rewards = []
    eval_success_rates = []
    eval_key_rates = []
    eval_door_rates = []
    eval_coverages = []
    eval_episodes = []
    
    # Create figure for visualization
    plt.figure(figsize=(12, 8))
    plt.ion()  # Interactive mode on
    
    # Training loop
    for episode in range(1, episodes + 1):
        # Setup for visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Episode {episode} Training')
        
        # Reset environment for new episode
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        option_changes = 0
        keys_collected = 0
        doors_opened = 0
        reward_pattern = []
        
        # Track previous option to detect changes
        prev_option = None
        
        # Get the base environment for rendering
        base_env = env.unwrapped
        
        # Start epsilon value for this episode
        if hasattr(dceo_agent, 'epsilon'):
            epsilon = dceo_agent.epsilon
            print(f"Starting Episode {episode} with epsilon {epsilon:.4f}")
        
        # Episode loop
        while not done:
            # Select and take action
            action = dceo_agent.select_action(obs)
            
            # Handle both gym interface versions
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
            
            # Update agent
            dceo_agent.step(obs, action, reward, next_obs, done)
            obs = next_obs
            
            # Track metrics
            steps += 1
            episode_reward += reward
            reward_pattern.append(round(reward, 1))
            if len(reward_pattern) > 5:
                reward_pattern.pop(0)
                
            # Check for option changes
            if hasattr(dceo_agent, 'current_option'):
                if dceo_agent.current_option != prev_option:
                    option_changes += 1
                    prev_option = dceo_agent.current_option
                    print(f"  Step {steps}: Switched to option {dceo_agent.current_option}")
            
            # Track key collection and door opening
            if info.get('key_collected', False):
                keys_collected += 1
                print(f"  Step {steps}: Collected key {keys_collected}")
                
            if info.get('door_opened', False):
                doors_opened += 1
                print(f"  Step {steps}: Opened door {doors_opened}")
            
            # Update epsilon for display
            if hasattr(dceo_agent, 'epsilon'):
                epsilon = dceo_agent.epsilon
            
            # Print progress periodically
            if steps % 50 == 0:
                print(f"  Step {steps}: Reward so far: {episode_reward:.2f}, Keys: {keys_collected}, Doors: {doors_opened}")
                print(f"  Current epsilon: {epsilon:.4f}")
                print(f"  Current option: {dceo_agent.current_option}")
                    
            # Visualize the environment state every few steps or on key events
            if steps % 5 == 0 or info.get('key_collected', False) or info.get('door_opened', False) or done:
                # Clear the axis and render the environment in first subplot
                ax1.clear()
                maze_img = base_env.render(mode='rgb_array')
                if maze_img is not None:
                    ax1.imshow(maze_img)
                    
                    # Display agent information
                    title = f"Episode {episode}, Step {steps}, Reward: {episode_reward:.2f}\n"
                    title += f"Keys: {keys_collected}/{env.unwrapped.num_keys}, "
                    title += f"Doors: {doors_opened}\n"
                    if hasattr(dceo_agent, 'current_option'):
                        title += f"Current Option: {dceo_agent.current_option}, Epsilon: {epsilon:.4f}"
                    ax1.set_title(title)
                
                # Plot training metrics in second subplot
                ax2.clear()
                ax2.plot(rewards, label='Training Rewards')
                if eval_rewards:
                    ax2.plot(eval_episodes, eval_rewards, 'r-', label='Eval Rewards')
                    ax2.plot(eval_episodes, eval_success_rates, 'g-', label='Success Rate')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Reward')
                ax2.legend()
                ax2.grid(True)
                
                # Show the plot
                display(fig)
                clear_output(wait=True)
        
        # Episode summary
        print(f"Episode {episode} done in {steps} steps")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Keys collected: {keys_collected}/{env.unwrapped.num_keys}")
        print(f"  Doors opened: {doors_opened}")
        print(f"  Goal reached: {info.get('goal_reached', False)}")
        print(f"  Option changes: {option_changes}")
        print(f"  Reward pattern: {', '.join([str(r) for r in reward_pattern])}... (last 5 steps)")
        if hasattr(dceo_agent, 'epsilon'):
            print(f"  Final epsilon: {epsilon:.4f}")
        
        # Store training results
        rewards.append(episode_reward)
        
        # Evaluate periodically
        if episode % eval_interval == 0:
            print(f"\nEvaluating after episode {episode}...")
            eval_results = evaluate_agent(dceo_agent, env, num_episodes=eval_episodes)
            print(f"  Eval reward: {eval_results['avg_reward']:.2f}")
            print(f"  Success rate: {eval_results['success_rate']:.2f}")
            print(f"  Keys collected: {eval_results['avg_keys']:.2f}/{env.unwrapped.num_keys}")
            print(f"  Doors opened: {eval_results['avg_doors']:.2f}")
            print(f"  Coverage: {eval_results['avg_coverage']:.2f}\n")
            
            # Store evaluation results
            eval_rewards.append(eval_results['avg_reward'])
            eval_success_rates.append(eval_results['success_rate'])
            eval_key_rates.append(eval_results['avg_keys'])
            eval_door_rates.append(eval_results['avg_doors'])
            eval_coverages.append(eval_results['avg_coverage'])
            eval_episodes.append(episode)
            
            # Save model checkpoint
            torch.save(dceo_agent.state_dict(), f'dceo_agent_checkpoint_ep{episode}.pth')
        
        plt.close(fig)
    
    # Final plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot training rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Plot evaluation rewards
    axes[0, 1].plot(eval_episodes, eval_rewards, 'r-')
    axes[0, 1].set_title('Evaluation Rewards')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True)
    
    # Plot success rate
    axes[1, 0].plot(eval_episodes, eval_success_rates, 'g-')
    axes[1, 0].set_title('Success Rate')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].grid(True)
    
    # Plot key and door collection
    axes[1, 1].plot(eval_episodes, eval_key_rates, 'y-', label='Keys')
    axes[1, 1].plot(eval_episodes, eval_door_rates, 'b-', label='Doors')
    axes[1, 1].set_title('Key and Door Collection Rates')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    
    print("Training complete! Results saved to 'training_results.png'")
    
    return {
        'train_rewards': rewards,
        'eval_rewards': eval_rewards,
        'eval_success_rates': eval_success_rates,
        'eval_key_rates': eval_key_rates,
        'eval_door_rates': eval_door_rates,
        'eval_coverages': eval_coverages,
        'eval_episodes': eval_episodes
    }

# Run the training with visualization
# results = visualize_and_train()

# Save the trained model
# torch.save(dceo_agent.state_dict(), 'dceo_agent_final.pth')
# print("Final model saved to 'dceo_agent_final.pth'")

# Save the results dictionary
# import pickle
# with open('training_results.pkl', 'wb') as f:
#     pickle.dump(results, f)
# print("Training results saved to 'training_results.pkl'")
