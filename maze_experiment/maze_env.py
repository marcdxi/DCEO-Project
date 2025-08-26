import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class MazeEnv(gym.Env):
    """
    A simple 8x8 maze environment.
    
    Actions:
    - 0: Move up
    - 1: Move right
    - 2: Move down
    - 3: Move left
    
    Rewards:
    - Reaching the goal: +1
    - Every step: -0.01 (small penalty to encourage efficient paths)
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, maze_size=8, max_steps=200, fixed_layout=False):
        """
        Initialize the maze environment.
        
        Args:
            maze_size: Size of the maze (maze_size x maze_size)
            max_steps: Maximum steps allowed before episode termination
            fixed_layout: Whether to use a fixed layout for the maze (same walls each reset)
        """
        super(MazeEnv, self).__init__()
        
        self.maze_size = maze_size
        self.max_steps = max_steps
        self.fixed_layout = fixed_layout
        self.maze = None
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = spaces.Box(low=0, high=1, 
                                           shape=(2,), dtype=np.float32)
        
        # Create a simple maze with walls
        self._create_maze()
        
        # Set start and goal positions
        self.start_pos = (0, 0)
        self.goal_pos = (maze_size - 1, maze_size - 1)
        
        # Initialize agent position
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        
        # For visualization
        self.fig = None
        self.ax = None
        
        # For tracking state coverage
        self.visited_states = set()
        
    def step(self, action):
        self.steps_taken += 1
        
        # Calculate new position based on action
        new_pos = list(self.agent_pos)
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Right
            new_pos[1] = min(self.maze_size - 1, new_pos[1] + 1)
        elif action == 2:  # Down
            new_pos[0] = min(self.maze_size - 1, new_pos[0] + 1)
        elif action == 3:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        
        # Check if the new position is valid (not a wall)
        new_pos = tuple(new_pos)
        if new_pos[0] < self.maze_size and new_pos[1] < self.maze_size and self.maze[new_pos] != 1:
            self.agent_pos = new_pos
        
        # Add to visited states
        self.visited_states.add(self.agent_pos)
        
        # Check if goal is reached
        done = self.agent_pos == self.goal_pos or self.steps_taken >= self.max_steps
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 1.0
        else:
            # Calculate Manhattan distance to goal
            dist_to_goal = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            max_dist = self.maze_size * 2  # Maximum possible Manhattan distance
            
            # Normalized distance-based reward component (higher when closer to goal)
            dist_reward = 0.1 * (1 - dist_to_goal / max_dist)
            
            # Small penalty for each step to encourage efficiency
            step_penalty = -0.01
            
            # Exploration bonus for visiting new states
            exploration_bonus = 0.05 if self.agent_pos not in self.visited_states else 0.0
            
            # Combine rewards
            reward = step_penalty + dist_reward + exploration_bonus
        
        # Create info dict
        info = {
            'steps': self.steps_taken,
            'position': self.agent_pos,
            'distance_to_goal': abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        }
        
        return self._get_obs(), reward, done, info
    
    def _create_maze(self):
        """Create the maze layout with walls."""
        self.maze = np.zeros((self.maze_size, self.maze_size))
        
        # Add some walls to make it interesting
        # Horizontal walls
        self.maze[2, 2:6] = 1
        self.maze[5, 3:7] = 1
        # Vertical walls
        self.maze[1:5, 3] = 1
        self.maze[4:7, 6] = 1
        
        # For larger mazes, add more complex structures
        if self.maze_size >= 12:
            # Additional horizontal walls
            self.maze[8, 4:10] = 1
            self.maze[10, 2:8] = 1
            # Additional vertical walls
            self.maze[8:12, 9] = 1
            self.maze[2:6, 8] = 1
    
    def reset(self):
        # If not using a fixed layout, recreate the maze
        if not self.fixed_layout:
            self._create_maze()
            
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.visited_states = set()  # Reset visited states
        return self._get_obs()
    
    def _get_obs(self):
        # Normalize position to [0, 1]
        return np.array([
            self.agent_pos[0] / (self.maze_size - 1),
            self.agent_pos[1] / (self.maze_size - 1)
        ], dtype=np.float32)
    
    def render(self, mode='human'):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            
        self.ax.clear()
        
        # Create a visualization of the maze
        maze_with_agent = self.maze.copy()
        maze_with_agent[self.agent_pos] = 2  # Agent
        maze_with_agent[self.goal_pos] = 3   # Goal
        maze_with_agent[self.start_pos] = 4  # Start
        
        # Create a colormap: 0=empty, 1=wall, 2=agent, 3=goal, 4=start
        cmap = ListedColormap(['white', 'black', 'red', 'green', 'blue'])
        
        # Plot the maze
        self.ax.imshow(maze_with_agent, cmap=cmap)
        
        # Add a grid
        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        self.ax.set_xticks(np.arange(-0.5, self.maze_size, 1))
        self.ax.set_yticks(np.arange(-0.5, self.maze_size, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Add a legend
        legend_elements = [
            mpatches.Patch(color='white', label='Empty'),
            mpatches.Patch(color='black', label='Wall'),
            mpatches.Patch(color='red', label='Agent'),
            mpatches.Patch(color='green', label='Goal'),
            mpatches.Patch(color='blue', label='Start')
        ]
        self.ax.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.05), ncol=5)
        
        self.ax.set_title(f'Step: {self.steps_taken}')
        
        if mode == 'human':
            plt.pause(0.1)
            return None
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def get_state_coverage(self):
        """Return the percentage of states visited"""
        total_states = (self.maze_size * self.maze_size) - np.sum(self.maze)  # Exclude walls
        return len(self.visited_states) / total_states