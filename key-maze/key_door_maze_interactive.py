"""
Modified Key-Door Maze Environment with interactive rendering.
This version updates the render function to use interactive mode.
"""

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import time

# Import the original environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from key_door_maze_env import KeyDoorMazeEnv

# Configure matplotlib for interactive mode
plt.ion()  # Enable interactive mode


class InteractiveKeyDoorMazeEnv(KeyDoorMazeEnv):
    """Interactive version of the Key-Door Maze environment with continuous rendering."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with the same parameters as the original environment."""
        super(InteractiveKeyDoorMazeEnv, self).__init__(*args, **kwargs)
        self.fig = None
        self.ax = None
        self.legend_added = False
    
    def render(self, mode='human'):
        """Render the current state of the environment with continuous updates."""
        # Create a colored image of the maze
        maze_img = np.zeros((self.maze_size, self.maze_size, 3))
        
        # Fill with cell colors
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if (i, j) == self.agent_position:
                    maze_img[i, j] = self.COLORS[self.AGENT]
                else:
                    cell_type = self.grid[i, j]
                    maze_img[i, j] = self.COLORS[cell_type]
        
        # Display using matplotlib
        if mode == 'human':
            if self.fig is None or not plt.fignum_exists(self.fig.number):
                self.fig, self.ax = plt.subplots(figsize=(6, 6))
                self.legend_added = False
                
            self.ax.clear()
            self.ax.imshow(maze_img)
            
            # Create legend if not already added
            if not self.legend_added:
                legend_elements = [
                    mpatches.Patch(color=self.COLORS[self.EMPTY], label='Empty'),
                    mpatches.Patch(color=self.COLORS[self.WALL], label='Wall'),
                    mpatches.Patch(color=self.COLORS[self.GOAL], label='Goal'),
                    mpatches.Patch(color=self.COLORS[self.KEY], label='Key'),
                    mpatches.Patch(color=self.COLORS[self.DOOR], label='Door'),
                    mpatches.Patch(color=self.COLORS[self.AGENT], label='Agent')
                ]
                self.ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
                self.legend_added = True
            
            # Display information about keys and doors
            keys_text = "Keys: " + " ".join(["Key" if k else "X" for k in self.inventory])
            doors_text = "Doors: " + " ".join(["Open" if d else "Locked" for d in self.door_status])
            steps_text = f"Steps: {self.steps_done}/{self.max_steps}"
            
            self.ax.set_title(f"{keys_text}  |  {doors_text}  |  {steps_text}")
            plt.tight_layout()
            
            # Update the display and pause briefly
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause to allow the display to update
            
        elif mode == 'rgb_array':
            return (maze_img * 255).astype(np.uint8)
    
    def close(self):
        """Close the environment and the rendering window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        super(InteractiveKeyDoorMazeEnv, self).close()
    
    def reset(self, seed=None, options=None):
        """Reset the environment and create a new maze."""
        # Reset the environment
        state = super(InteractiveKeyDoorMazeEnv, self).reset(seed=seed, options=options)
        
        # Close and reset the rendering window
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        return state
