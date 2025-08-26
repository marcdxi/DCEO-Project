"""
Visualization script for Key-Door Maze environment.
Provides a clear visual representation of the maze, agent, keys, doors, and goal.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from key_door_maze import KeyDoorMaze
import time

def visualize_maze(env, title="Key-Door Maze"):
    """Visualize the current state of the maze"""
    # Create a grid representation for visualization
    grid = np.zeros((env.size, env.size, 3))  # RGB grid
    
    # Define colors
    WALL_COLOR = [0.3, 0.3, 0.3]  # Dark gray
    AGENT_COLOR = [1.0, 0.0, 0.0]  # Red
    GOAL_COLOR = [0.0, 1.0, 0.0]  # Green
    FLOOR_COLOR = [1.0, 1.0, 1.0]  # White
    VISITED_COLOR = [0.9, 0.9, 0.9]  # Light gray
    
    # Key colors (different colors for different keys)
    KEY_COLORS = [
        [1.0, 1.0, 0.0],  # Yellow
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Purple
    ]
    
    # Door colors (matching their key colors but slightly darker)
    DOOR_COLORS = [
        [0.8, 0.8, 0.0],  # Dark yellow
        [0.0, 0.8, 0.8],  # Dark cyan
        [0.8, 0.4, 0.0],  # Dark orange
        [0.4, 0.0, 0.8],  # Dark purple
    ]
    
    # Set the floor color as default
    for i in range(env.size):
        for j in range(env.size):
            grid[i, j] = FLOOR_COLOR
    
    # Mark visited positions
    for pos in env.visited_positions:
        i, j = pos
        grid[i, j] = VISITED_COLOR
    
    # Add maze elements
    for i in range(env.size):
        for j in range(env.size):
            cell_type = env.grid[i, j]
            
            # Walls
            if cell_type == env.WALL:
                grid[i, j] = WALL_COLOR
            
            # Keys
            elif env.KEY <= cell_type < env.DOOR:
                key_id = cell_type - env.KEY
                grid[i, j] = KEY_COLORS[key_id % len(KEY_COLORS)]
            
            # Doors
            elif env.DOOR <= cell_type < env.GOAL:
                door_id = cell_type - env.DOOR
                # Check if door is unlocked
                if env.inventory[door_id]:
                    # Unlocked door - lighter color
                    grid[i, j] = [c + 0.2 for c in DOOR_COLORS[door_id % len(DOOR_COLORS)]]
                else:
                    # Locked door
                    grid[i, j] = DOOR_COLORS[door_id % len(DOOR_COLORS)]
            
            # Goal
            elif cell_type == env.GOAL:
                grid[i, j] = GOAL_COLOR
    
    # Add agent
    i, j = env.agent_position
    grid[i, j] = AGENT_COLOR
    
    # Create the figure and display the maze
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    
    # Add grid lines
    plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    plt.xticks(np.arange(-.5, env.size, 1), [])
    plt.yticks(np.arange(-.5, env.size, 1), [])
    
    # Add inventory display
    inventory_text = "Inventory: "
    for i in range(env.num_keys):
        if env.inventory[i]:
            key_color = KEY_COLORS[i % len(KEY_COLORS)]
            hex_color = '#%02x%02x%02x' % tuple(int(c*255) for c in key_color)
            inventory_text += f"<Key {i+1}> "
    
    plt.figtext(0.5, 0.01, inventory_text, ha="center", fontsize=12)
    
    # Add title with info
    title_text = f"{title}\nStep: {env.current_step}, Keys: {sum(env.inventory)}/{env.num_keys}"
    plt.title(title_text)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=AGENT_COLOR, label='Agent'),
        plt.Rectangle((0, 0), 1, 1, color=WALL_COLOR, label='Wall'),
        plt.Rectangle((0, 0), 1, 1, color=GOAL_COLOR, label='Goal'),
        plt.Rectangle((0, 0), 1, 1, color=VISITED_COLOR, label='Visited')
    ]
    
    # Add key and door colors to legend
    for i in range(min(env.num_keys, len(KEY_COLORS))):
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color=KEY_COLORS[i], label=f'Key {i+1}')
        )
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color=DOOR_COLORS[i], label=f'Door {i+1}')
        )
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig('maze_visualization.png', dpi=100, bbox_inches='tight')
    plt.close()  # Close the figure to prevent display in notebooks
    
    return 'maze_visualization.png'

def animate_episode(maze_size=10, num_keys=2, max_steps=200, seed=42):
    """Run and visualize an episode with a random policy"""
    # Create environment
    env = KeyDoorMaze(size=maze_size, num_keys=num_keys, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    
    done = False
    truncated = False
    step = 0
    total_reward = 0
    frames = []
    
    # Save initial state
    img_path = visualize_maze(env, title=f"Key-Door Maze - Initial State")
    frames.append(img_path)
    
    # Run random policy
    while not (done or truncated) and step < max_steps:
        # Take random action
        action = random.randint(0, 3)
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        # Generate visualization
        title = f"Key-Door Maze - Step {step}, Reward: {total_reward:.2f}"
        if done:
            title += " (Success!)"
        elif truncated:
            title += " (Truncated)"
        
        img_path = visualize_maze(env, title=title)
        frames.append(img_path)
    
    print(f"Episode completed in {step} steps with reward {total_reward}")
    return frames

if __name__ == "__main__":
    # Create and visualize a static maze
    maze_size = 10
    num_keys = 2
    max_steps = 200
    seed = 42
    
    env = KeyDoorMaze(size=maze_size, num_keys=num_keys, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    
    # Visualize initial state
    img_path = visualize_maze(env, title="Key-Door Maze - Initial State")
    print(f"Maze visualization saved to {img_path}")
    
    # For animation, uncomment the following:
    # frames = animate_episode(maze_size, num_keys, max_steps, seed)
    # print(f"Generated {len(frames)} frames")
