"""
Complex Maze Environment for Reinforcement Learning.

This environment implements a more complex maze with an agent, walls, and a goal.
The agent must navigate through the maze to reach the goal.
"""

import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import random
from collections import deque

class ComplexMazeEnv(gym.Env):
    """
    A Complex Maze environment compatible with Rainbow DCEO agents.
    
    The maze contains walls and a goal. The agent must navigate
    through the maze to reach the goal.
    
    Actions:
    - 0: Move up
    - 1: Move right
    - 2: Move down
    - 3: Move left
    
    Rewards:
    - Reaching the goal: +10.0
    - Every step: -0.01 (small penalty to encourage efficient paths)
    - Exploration bonus: +0.05 for visiting a new cell
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    # Cell types
    EMPTY = 0
    WALL = 1
    GOAL = 2
    AGENT = 3
    
    # Colors for rendering
    COLORS = {
        EMPTY: [1.0, 1.0, 1.0],  # White
        WALL: [0.0, 0.0, 0.0],   # Black
        GOAL: [0.0, 1.0, 0.0],   # Green
        AGENT: [1.0, 0.0, 0.0]   # Red
    }
    
    def __init__(self, maze_size=15, max_steps=200, use_fixed_layout=False, use_fixed_seed=False, fixed_seed=42):
        """Initialize the Complex Maze environment.
        
        Args:
            maze_size: Size of the maze grid (maze_size x maze_size)
            max_steps: Maximum steps per episode
            use_fixed_seed: Whether to use a fixed seed for maze generation
            fixed_seed: The fixed seed to use
            use_fixed_layout: Whether to reuse maze layout across episodes
        """
        super(ComplexMazeEnv, self).__init__()
        
        self.maze_size = maze_size
        self.max_steps = max_steps
        self.use_fixed_seed = use_fixed_seed
        self.fixed_seed = fixed_seed
        self.use_fixed_layout = use_fixed_layout
        
        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
        
        # Define observation space: multi-channel grid representation
        # Channel 0: Agent location
        # Channel 1: Walls
        # Channel 2: Goal
        self.channels = 3
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(maze_size, maze_size, self.channels), 
            dtype=np.float32
        )
        
        # Initialize maze
        self.generate_maze()
        
        # Additional tracking variables
        self.steps_done = 0
        self.visited_positions = set()
        
        # For visualization
        self.fig = None
        self.ax = None
        
    def seed(self, seed=None):
        """Set the random seed for this environment."""
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
        
    def generate_maze(self):
        """Generate a new maze with complex structure but ensuring path to goal."""
        if self.use_fixed_seed and self.fixed_seed is not None:
            # Use fixed seed for reproducible maze layouts
            np.random.seed(self.fixed_seed)
            
        max_attempts = 10  # Maximum attempts to generate a valid maze
        for attempt in range(max_attempts):
            # Create an empty grid
            self.grid = np.zeros((self.maze_size, self.maze_size), dtype=np.uint8)
            
            # Add maze boundary
            self._add_outer_walls()
            
            # Place agent in the top-left corner
            self.agent_position = (1, 1)  # Top-left corner, inside the outer wall
            
            # Place goal in the bottom-right corner
            self.goal_position = (self.maze_size-2, self.maze_size-2)  # Bottom-right corner, inside the outer wall
            self.grid[self.goal_position] = self.GOAL
            
            # Add more internal walls (increased density)
            self._add_complex_internal_walls()
            
            # Verify the maze is navigable
            if self._is_maze_navigable():
                print(f"Generated valid maze after {attempt+1} attempts")
                break
            else:
                print(f"Attempt {attempt+1}: Generated maze is not navigable, regenerating...")
            
            # Additionally verify the maze has the desired wall percentage
            wall_percentage = np.sum(self.grid == self.WALL) / (self.maze_size * self.maze_size)
            if wall_percentage < 0.5:  # Less than 50% walls
                print(f"Attempt {attempt+1}: Not enough walls ({wall_percentage:.2f}), regenerating...")
        
        # If we exhausted all attempts, create a simple maze with a valid path
        if attempt == max_attempts - 1:
            print("Exhausted maximum attempts. Creating simplified maze...")
            self._create_simple_maze()
        
        # Reset tracking variables
        self.steps_done = 0
        self.visited_positions = {self.agent_position}
        
    def _add_outer_walls(self):
        """Add walls around the perimeter of the maze."""
        # Top and bottom walls
        self.grid[0, :] = self.WALL
        self.grid[self.maze_size-1, :] = self.WALL
        
        # Left and right walls
        self.grid[:, 0] = self.WALL
        self.grid[:, self.maze_size-1] = self.WALL
        
    def _add_complex_internal_walls(self):
        """Generate a maze with complex corridors, multiple paths and challenges."""
        # Start with a completely filled grid (except boundaries)
        self.grid = np.ones((self.maze_size, self.maze_size), dtype=np.uint8)
        self._add_outer_walls()  # Add walls around the perimeter
        
        # Place agent and goal positions
        # For more challenging mazes, agent starts in top-left area and goal in bottom-right
        if self.maze_size >= 15:
            self.agent_position = (random.randint(1, 3), random.randint(1, 3))
            self.goal_position = (random.randint(self.maze_size-4, self.maze_size-2), 
                               random.randint(self.maze_size-4, self.maze_size-2))
        else:
            self.agent_position = (1, 1)  # Top-left corner, inside the outer wall
            self.goal_position = (self.maze_size-2, self.maze_size-2)  # Bottom-right
        
        # Mark agent and goal positions
        self.grid[self.agent_position] = self.EMPTY
        self.grid[self.goal_position] = self.GOAL
        
        # Generate initial maze structure using recursive backtracking
        # This guarantees a fully connected maze with exactly one path between any two points
        self._carve_maze_from_point(self.agent_position)
        
        # Add additional corridors and loops to create multiple paths
        self._add_maze_loops(num_loops=max(5, self.maze_size // 3))
        
        # Create a more direct path to the goal
        self._create_somewhat_direct_path(self.agent_position, self.goal_position)
        
        # Add some additional corridor branches to increase complexity
        self._add_random_corridor_branches()
        
        # Final verification that all areas are accessible
        self._ensure_all_corridors_accessible()
        
        # Make sure the agent position and goal are properly marked
        self.grid[self.agent_position] = self.EMPTY
        self.grid[self.goal_position] = self.GOAL
    
    def _recursive_division(self, x1, y1, x2, y2, depth=0):
        """Create a maze using recursive division method.
        This generates wall patterns similar to the reference image."""
        # Base case: if the area is too small, stop
        if x2 - x1 < 3 or y2 - y1 < 3:
            return
        
        # Create open corridors for both agent and goal
        self.grid[self.agent_position] = self.EMPTY
        self.grid[self.goal_position] = self.GOAL
        
        # Set all other non-boundary cells to empty
        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                self.grid[x, y] = self.EMPTY
        
        # Choose orientation based on dimensions
        # This creates more balanced divisions
        horizontal = random.random() > 0.5
        if x2 - x1 < y2 - y1:
            horizontal = True
        elif y2 - y1 < x2 - x1:
            horizontal = False
        
        # Maximum depth to avoid too complex mazes
        if depth > 5:
            return
        
        if horizontal:
            # Horizontal division
            wall_y = random.randint(y1 + 1, y2 - 1)
            passage_x = random.randint(x1, x2)
            
            # Create horizontal wall with one passage
            for x in range(x1, x2+1):
                if x != passage_x:
                    self.grid[x, wall_y] = self.WALL
            
            # Continue recursively dividing subregions
            self._recursive_division(x1, y1, x2, wall_y-1, depth+1)
            self._recursive_division(x1, wall_y+1, x2, y2, depth+1)
        else:
            # Vertical division
            wall_x = random.randint(x1 + 1, x2 - 1)
            passage_y = random.randint(y1, y2)
            
            # Create vertical wall with one passage
            for y in range(y1, y2+1):
                if y != passage_y:
                    self.grid[wall_x, y] = self.WALL
            
            # Continue recursively dividing subregions
            self._recursive_division(x1, y1, wall_x-1, y2, depth+1)
            self._recursive_division(wall_x+1, y1, x2, y2, depth+1)
    
    def _create_winding_path_to_goal(self):
        """Create a non-direct, winding path from start to goal."""
        # Use a random walk approach with a bias toward the goal
        current = self.agent_position
        goal = self.goal_position
        visited = {current}
        
        # Clear the starting position
        self.grid[current] = self.EMPTY
        
        # Directions: Up, Right, Down, Left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Create a winding path with detours
        max_detour_length = max(3, self.maze_size // 4)  # Larger detours for bigger mazes
        goal_bias = 0.7  # 70% chance to move toward goal, 30% chance for random direction
        
        # Calculate Manhattan distance to goal for heuristic
        def manhattan_dist(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # Find next direction that moves closer to the goal
        def get_goal_directed_directions(current_pos):
            # Sort directions to prefer those that move closer to goal
            return sorted(directions, key=lambda d: manhattan_dist(
                (current_pos[0] + d[0], current_pos[1] + d[1]), goal))
        
        # While we haven't reached the goal
        detour_steps = 0
        in_detour = False
        detour_direction = None
        
        while current != goal:
            if in_detour and detour_steps > 0:
                # Continue the detour
                possible_dirs = [d for d in directions if 
                              1 <= current[0] + d[0] < self.maze_size - 1 and 
                              1 <= current[1] + d[1] < self.maze_size - 1]
                if possible_dirs:
                    direction = detour_direction or random.choice(possible_dirs)
                    detour_steps -= 1
                else:
                    # End detour if no valid directions
                    in_detour = False
                    detour_steps = 0
                    continue
            elif random.random() < 0.15 and not in_detour:  # Start a new detour
                in_detour = True
                detour_steps = random.randint(2, max_detour_length)
                possible_dirs = [d for d in directions if 
                              1 <= current[0] + d[0] < self.maze_size - 1 and 
                              1 <= current[1] + d[1] < self.maze_size - 1]
                if possible_dirs:
                    detour_direction = random.choice(possible_dirs)
                    direction = detour_direction
                else:
                    # Skip detour if no valid directions
                    in_detour = False
                    detour_steps = 0
                    continue
            else:  # Move toward goal with some randomness
                if random.random() < goal_bias:
                    # Use goal-directed movement
                    sorted_dirs = get_goal_directed_directions(current)
                    direction = sorted_dirs[0]  # Best direction toward goal
                else:
                    # Use random direction for exploration
                    possible_dirs = [d for d in directions if 
                                  1 <= current[0] + d[0] < self.maze_size - 1 and 
                                  1 <= current[1] + d[1] < self.maze_size - 1]
                    if possible_dirs:
                        direction = random.choice(possible_dirs)
                    else:
                        # If no valid directions, use goal-directed
                        sorted_dirs = get_goal_directed_directions(current)
                        direction = sorted_dirs[0] 
            
            # Calculate next position
            next_pos = (current[0] + direction[0], current[1] + direction[1])
            
            # Check bounds
            if (1 <= next_pos[0] < self.maze_size - 1 and 
                1 <= next_pos[1] < self.maze_size - 1):
                
                # Clear the path
                self.grid[next_pos] = self.EMPTY
                visited.add(next_pos)
                current = next_pos
                
                # If we're close to the goal, increase bias to reach it
                if manhattan_dist(current, goal) <= 3:
                    goal_bias = 0.9  # Stronger pull toward goal when close
            else:
                # Hit a boundary, end any detour
                in_detour = False
                detour_steps = 0
        
        # Make sure goal is properly marked
        self.grid[goal] = self.GOAL
        
        return visited
    
    def _recursive_division_with_loops(self, x1, y1, x2, y2, depth=0):
        """Enhanced recursive division that creates multiple paths and loops."""
        # Base case: if the area is too small, stop
        if x2 - x1 < 3 or y2 - y1 < 3:
            return
        
        # Set a maximum recursion depth
        if depth > 8:  # Deeper recursion for more complex mazes
            return
        
        # Ensure agent and goal positions are clear
        self.grid[self.agent_position] = self.EMPTY
        self.grid[self.goal_position] = self.GOAL
        
        # Choose orientation based on dimensions but with randomness
        # This creates more varied maze layouts
        if x2 - x1 < y2 - y1:
            horizontal = True
        elif y2 - y1 < x2 - x1:
            horizontal = False
        else:
            horizontal = random.random() > 0.5
        
        if horizontal:
            # Horizontal division
            wall_y = random.randint(y1 + 1, y2 - 1)
            
            # Create multiple passages (typically 2-3) for more route options
            num_passages = random.randint(2, max(2, min(4, (x2 - x1) // 3)))
            passages = sorted(random.sample(range(x1, x2+1), num_passages))
            
            # Create horizontal wall with multiple passages
            for x in range(x1, x2+1):
                if x not in passages:
                    self.grid[x, wall_y] = self.WALL
            
            # Continue recursively dividing subregions
            self._recursive_division_with_loops(x1, y1, x2, wall_y-1, depth+1)
            self._recursive_division_with_loops(x1, wall_y+1, x2, y2, depth+1)
        else:
            # Vertical division
            wall_x = random.randint(x1 + 1, x2 - 1)
            
            # Create multiple passages for more route options
            num_passages = random.randint(2, max(2, min(4, (y2 - y1) // 3)))
            passages = sorted(random.sample(range(y1, y2+1), num_passages))
            
            # Create vertical wall with multiple passages
            for y in range(y1, y2+1):
                if y not in passages:
                    self.grid[wall_x, y] = self.WALL
            
            # Continue recursively dividing subregions
            self._recursive_division_with_loops(x1, y1, wall_x-1, y2, depth+1)
            self._recursive_division_with_loops(wall_x+1, y1, x2, y2, depth+1)
        
        # Randomly skip some divisions to create more open areas (0-20% chance)
        if random.random() < 0.15 and depth > 2:
            # Create a small room or junction by clearing an area
            room_x1 = random.randint(x1, x2-2)
            room_y1 = random.randint(y1, y2-2)
            room_size = random.randint(2, min(4, min(x2-room_x1, y2-room_y1)))
            
            for rx in range(room_x1, min(x2, room_x1+room_size)):
                for ry in range(room_y1, min(y2, room_y1+room_size)):
                    if (rx, ry) != self.agent_position and (rx, ry) != self.goal_position:
                        self.grid[rx, ry] = self.EMPTY
    
    def _add_corridor_branches(self):
        """Add additional corridor branches to create a more complex maze structure."""
        empty_cells = [(r, c) for r in range(1, self.maze_size-1) 
                      for c in range(1, self.maze_size-1) 
                      if self.grid[r, c] == self.EMPTY and
                      (r, c) != self.agent_position and
                      (r, c) != self.goal_position]
        
        if not empty_cells:
            return
        
        # Calculate number of branches based on maze size
        num_branches = max(5, min(15, self.maze_size // 2))
        branch_attempts = min(len(empty_cells) // 2, num_branches * 3)  # More attempts than branches
        
        branches_created = 0
        for _ in range(branch_attempts):
            if branches_created >= num_branches or not empty_cells:
                break
                
            # Select a random cell to start a branch
            start_idx = random.randint(0, len(empty_cells) - 1)
            start_cell = empty_cells[start_idx]
            empty_cells.pop(start_idx)
            
            # Look for nearby walls to create branches from
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            # Shuffle directions for randomness
            random.shuffle(directions)
            
            for d_r, d_c in directions:
                # Determine branch length and direction
                branch_length = random.randint(2, max(3, self.maze_size // 5))
                
                # Create the branch
                current_r, current_c = start_cell
                branch_cells = []
                
                for i in range(branch_length):
                    new_r = current_r + d_r
                    new_c = current_c + d_c
                    
                    # Check bounds and avoid crossing paths
                    if (1 <= new_r < self.maze_size-1 and 
                        1 <= new_c < self.maze_size-1 and
                        self.grid[new_r, new_c] == self.WALL):
                        
                        # Clear the cell
                        self.grid[new_r, new_c] = self.EMPTY
                        branch_cells.append((new_r, new_c))
                        current_r, current_c = new_r, new_c
                    else:
                        break
                
                # If we created a substantial branch, count it
                if len(branch_cells) >= 2:
                    branches_created += 1
                    break  # Move to next start cell
        
        # Connect some branches to create loops (about 30% of branches)
        self._create_maze_loops(num_loops=max(2, branches_created // 3))
    
    def _create_maze_loops(self, num_loops=3):
        """Create loops in the maze by connecting nearby corridors."""
        for _ in range(num_loops):
            # Find empty cells that have at least two adjacent walls
            empty_cells = [(r, c) for r in range(1, self.maze_size-1) 
                          for c in range(1, self.maze_size-1) 
                          if self.grid[r, c] == self.EMPTY]
            
            if not empty_cells:
                return
                
            random.shuffle(empty_cells)
            
            for cell in empty_cells:
                r, c = cell
                # Directions: Up, Right, Down, Left
                directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                random.shuffle(directions)
                
                # Look for a wall with empty space on the other side
                for d_r, d_c in directions:
                    # Wall position and position beyond the wall
                    wall_r, wall_c = r + d_r, c + d_c
                    beyond_r, beyond_c = r + 2*d_r, c + 2*d_c
                    
                    # Check if wall exists and beyond is empty (potential loop)
                    if (1 <= wall_r < self.maze_size-1 and 
                        1 <= wall_c < self.maze_size-1 and
                        1 <= beyond_r < self.maze_size-1 and 
                        1 <= beyond_c < self.maze_size-1 and
                        self.grid[wall_r, wall_c] == self.WALL and
                        self.grid[beyond_r, beyond_c] == self.EMPTY):
                        
                        # Break through the wall to create a loop
                        self.grid[wall_r, wall_c] = self.EMPTY
                        break  # Only create one loop from this cell
                
                # Move to next potential loop after creating one
                if self.grid[wall_r, wall_c] == self.EMPTY:
                    break
    
    def _add_random_corridor_structure(self):
        """Create a more maze-like structure with narrow corridors throughout."""
        # Start by filling the maze with walls
        for r in range(1, self.maze_size-1):
            for c in range(1, self.maze_size-1):
                if (r, c) != self.agent_position and (r, c) != self.goal_position:
                    self.grid[r, c] = self.WALL
        
        # Now we'll carve corridors using a modified recursive backtracking algorithm
        # This creates the classic maze-like structure with narrow corridors
        
        # Start from the agent position
        self._carve_maze_from_point(self.agent_position)
        
        # Make sure the goal is reachable
        if not self._is_maze_navigable():
            self._create_path_to_goal()
            
        # Add some loops to create multiple paths
        self._add_maze_loops()
    
    def _connect_points_with_corridor(self, point1, point2):
        """Connect two points with a corridor that has some randomness."""
        r1, c1 = point1
        r2, c2 = point2
        
        # Current position
        r, c = r1, c1
        
        # Create a winding path with 30% chance of a random direction at each step
        while (r, c) != (r2, c2):
            # Mark current position as empty
            self.grid[r, c] = self.EMPTY
            
            # Determine direction with bias toward destination
            if random.random() < 0.7:  # 70% chance to move toward destination
                # Choose direction that gets us closer to destination
                if r < r2:
                    dr, dc = 1, 0  # Move down
                elif r > r2:
                    dr, dc = -1, 0  # Move up
                elif c < c2:
                    dr, dc = 0, 1  # Move right
                else:  # c > c2
                    dr, dc = 0, -1  # Move left
            else:  # 30% chance for random direction (adds windiness)
                # Random direction that's not going backwards
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                dr, dc = random.choice(directions)
            
            # Calculate new position
            new_r, new_c = r + dr, c + dc
            
            # Check bounds
            if 1 <= new_r < self.maze_size-1 and 1 <= new_c < self.maze_size-1:
                r, c = new_r, new_c
            # If out of bounds, try a different direction next time
    
    def _add_strategic_obstacles(self):
        """Add strategic obstacles to make the maze more challenging."""
        # In the new maze generation approach, we don't need as many additional obstacles
        # since we're starting with a full maze and carving paths
        # Instead, we'll just add a few strategic blocks at junctions
        
        # Find junction points (cells with 3+ adjacent empty cells)
        junction_points = []
        
        for r in range(1, self.maze_size-1):
            for c in range(1, self.maze_size-1):
                if self.grid[r, c] == self.EMPTY and (r, c) != self.agent_position and (r, c) != self.goal_position:
                    # Count adjacent empty cells
                    adjacent_empty = 0
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if (0 <= r+dr < self.maze_size and 0 <= c+dc < self.maze_size and 
                            self.grid[r+dr, c+dc] == self.EMPTY):
                            adjacent_empty += 1
                    
                    if adjacent_empty >= 3:  # This is a junction with 3+ connections
                        junction_points.append((r, c))
        
        # Add some obstacles at random junctions (about 20% of them)
        num_obstacles = min(len(junction_points) // 5, 5)
        if junction_points and num_obstacles > 0:
            for pos in random.sample(junction_points, num_obstacles):
                r, c = pos
                # Only add if it won't disconnect the maze
                self.grid[r, c] = self.WALL
                if not self._is_maze_navigable():
                    # If it disconnects the maze, revert
                    self.grid[r, c] = self.EMPTY
    
    def _add_narrow_passages(self):
        """Add narrow passages to create bottlenecks in the maze."""
        # Find longer empty horizontal and vertical lines to add bottlenecks
        min_line_length = 5  # Minimum length of corridor to consider
        
        # Check horizontal lines
        for r in range(1, self.maze_size-1):
            empty_count = 0
            start_c = -1
            
            for c in range(1, self.maze_size-1):
                if self.grid[r, c] == self.EMPTY:
                    if empty_count == 0:
                        start_c = c
                    empty_count += 1
                else:
                    if empty_count >= min_line_length:
                        # Found a horizontal line, consider adding a barrier
                        mid_c = start_c + empty_count // 2
                        if (r, mid_c) != self.agent_position and (r, mid_c) != self.goal_position:
                            # Check that there's a path around this barrier
                            # We only place if there are empty cells above or below
                            if ((r > 1 and self.grid[r-1, mid_c] == self.EMPTY) or
                                (r < self.maze_size-2 and self.grid[r+1, mid_c] == self.EMPTY)):
                                if random.random() < 0.4:  # 40% chance to add a barrier
                                    self.grid[r, mid_c] = self.WALL
                    
                    empty_count = 0
        
        # Check vertical lines
        for c in range(1, self.maze_size-1):
            empty_count = 0
            start_r = -1
            
            for r in range(1, self.maze_size-1):
                if self.grid[r, c] == self.EMPTY:
                    if empty_count == 0:
                        start_r = r
                    empty_count += 1
                else:
                    if empty_count >= min_line_length:
                        # Found a vertical line, consider adding a barrier
                        mid_r = start_r + empty_count // 2
                        if (mid_r, c) != self.agent_position and (mid_r, c) != self.goal_position:
                            # Check that there's a path around this barrier
                            # We only place if there are empty cells to the left or right
                            if ((c > 1 and self.grid[mid_r, c-1] == self.EMPTY) or
                                (c < self.maze_size-2 and self.grid[mid_r, c+1] == self.EMPTY)):
                                if random.random() < 0.4:  # 40% chance to add a barrier
                                    self.grid[mid_r, c] = self.WALL
                    
                    empty_count = 0
    
    def _carve_maze_from_point(self, start_point):
        """Use recursive backtracking to carve a maze starting from a point."""
        # Directions: Up, Right, Down, Left
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        
        # Stack for backtracking
        stack = [start_point]
        visited = {start_point}
        
        # Mark start as empty
        self.grid[start_point] = self.EMPTY
        
        while stack:
            r, c = stack[-1]
            
            # Find unvisited neighbors (with a wall in between)
            unvisited_neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check if the neighbor is within bounds and unvisited
                if (1 <= nr < self.maze_size-1 and 1 <= nc < self.maze_size-1 and 
                    (nr, nc) not in visited):
                    unvisited_neighbors.append((nr, nc, dr, dc))
            
            # If there are unvisited neighbors, pick one randomly
            if unvisited_neighbors:
                # Choose a random neighbor
                nr, nc, dr, dc = random.choice(unvisited_neighbors)
                
                # Remove the wall between current cell and chosen neighbor
                wall_r, wall_c = r + dr//2, c + dc//2
                self.grid[wall_r, wall_c] = self.EMPTY
                
                # Mark the neighbor as empty and visited
                self.grid[nr, nc] = self.EMPTY
                visited.add((nr, nc))
                
                # Push the neighbor onto the stack
                stack.append((nr, nc))
            else:
                # No unvisited neighbors, backtrack
                stack.pop()
                
        # Make sure the goal is clear
        self.grid[self.goal_position] = self.GOAL
    
    def _add_maze_loops(self, num_loops=None):
        """Add loops to the maze to create alternative paths.
        
        Args:
            num_loops: Number of loops to add. If None, will be calculated based on maze size.
        """
        # Number of loops to add (based on maze size if not specified)
        if num_loops is None:
            num_loops = max(2, min(8, self.maze_size // 3))
        
        # Find walls that could potentially be removed to create loops
        potential_loop_walls = []
        
        # Consider only walls that are not on the boundary
        for r in range(1, self.maze_size-1):
            for c in range(1, self.maze_size-1):
                if self.grid[r, c] == self.WALL:
                    # Check if removing this wall would create a loop
                    # A wall creates a loop if it connects two corridor cells
                    # that are already connected through the maze
                    
                    # Check horizontal connections
                    if (c > 1 and c < self.maze_size-2 and 
                        self.grid[r, c-1] == self.EMPTY and 
                        self.grid[r, c+1] == self.EMPTY):
                        potential_loop_walls.append((r, c))
                    
                    # Check vertical connections
                    elif (r > 1 and r < self.maze_size-2 and 
                          self.grid[r-1, c] == self.EMPTY and 
                          self.grid[r+1, c] == self.EMPTY):
                        potential_loop_walls.append((r, c))
        
        # Shuffle the list of potential walls
        random.shuffle(potential_loop_walls)
        
        # Remove some walls to create loops
        loops_created = 0
        for wall_pos in potential_loop_walls:
            if loops_created >= num_loops:
                break
                
            r, c = wall_pos
            
            # Remove the wall
            self.grid[r, c] = self.EMPTY
            loops_created += 1
        
        # Make sure agent and goal are properly set
        self.grid[self.agent_position] = self.EMPTY
        self.grid[self.goal_position] = self.GOAL
        
        return loops_created
    
    def _add_random_corridor_branches(self):
        """Add additional corridor branches to increase maze complexity."""
        # Number of branches to add (based on maze size)
        num_branches = max(3, min(8, self.maze_size // 2))
        
        # Find potential points to start new branches
        potential_branch_points = []
        
        # Look for dead ends (corridor cells with 3 adjacent walls)
        for r in range(1, self.maze_size-1):
            for c in range(1, self.maze_size-1):
                if self.grid[r, c] == self.EMPTY:
                    # Count adjacent walls
                    wall_count = 0
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if (0 <= r+dr < self.maze_size and 0 <= c+dc < self.maze_size and 
                            self.grid[r+dr, c+dc] == self.WALL):
                            wall_count += 1
                    
                    # If this is a dead end, add it as a potential branch point
                    if wall_count == 3:
                        potential_branch_points.append((r, c))
        
        # Shuffle the branch points
        random.shuffle(potential_branch_points)
        
        # Add branches from some dead ends
        branches_added = 0
        for r, c in potential_branch_points:
            if branches_added >= num_branches:
                break
                
            # Find the walled directions
            walled_directions = []
            for i, (dr, dc) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
                if (0 <= r+dr < self.maze_size and 0 <= c+dc < self.maze_size and 
                    self.grid[r+dr, c+dc] == self.WALL):
                    walled_directions.append((dr, dc))
            
            # Choose a random walled direction to break through
            if walled_directions:
                dr, dc = random.choice(walled_directions)
                
                # Choose branch length (2-4 cells)
                branch_length = random.randint(2, min(4, self.maze_size // 3))
                
                # Create branch
                curr_r, curr_c = r, c
                for _ in range(branch_length):
                    new_r, new_c = curr_r + dr, curr_c + dc
                    
                    # Check bounds
                    if 1 <= new_r < self.maze_size-1 and 1 <= new_c < self.maze_size-1:
                        # Carve corridor
                        self.grid[new_r, new_c] = self.EMPTY
                        curr_r, curr_c = new_r, new_c
                    else:
                        break
                
                branches_added += 1
        
        return branches_added
    
    def _ensure_all_corridors_accessible(self):
        """Make sure all corridor cells are accessible from the agent's position."""
        # Use BFS to find all cells accessible from the agent
        accessible = set()
        
        # Queue for BFS
        queue = deque([self.agent_position])
        accessible.add(self.agent_position)
        
        while queue:
            r, c = queue.popleft()
            
            # Check all four adjacent cells
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                # Check if neighbor is valid and unvisited
                if (0 <= nr < self.maze_size and 0 <= nc < self.maze_size and 
                    self.grid[nr, nc] != self.WALL and (nr, nc) not in accessible):
                    accessible.add((nr, nc))
                    queue.append((nr, nc))
        
        # Find all empty cells
        empty_cells = set()
        for r in range(self.maze_size):
            for c in range(self.maze_size):
                if self.grid[r, c] != self.WALL:
                    empty_cells.add((r, c))
        
        # Find inaccessible cells
        inaccessible = empty_cells - accessible
        
        # If there are inaccessible cells, connect them to the main maze
        if inaccessible:
            # For each inaccessible region, connect it to the main maze
            while inaccessible:
                # Find an inaccessible cell
                start_cell = next(iter(inaccessible))
                
                # Find all cells in this inaccessible region
                region = {start_cell}
                queue = deque([start_cell])
                
                while queue:
                    r, c = queue.popleft()
                    
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        
                        if ((nr, nc) in inaccessible and (nr, nc) not in region):
                            region.add((nr, nc))
                            queue.append((nr, nc))
                
                # Connect this region to the main maze by finding the shortest path
                min_dist = float('inf')
                best_connection = None
                
                for r1, c1 in region:
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        r2, c2 = r1 + dr, c1 + dc
                        
                        # Check if we're connecting to an accessible cell
                        if (r2, c2) in accessible:
                            best_connection = ((r1, c1), (r2, c2))
                            min_dist = 1  # Direct connection is the best
                            break
                        
                        # Check if there's a wall that could be removed
                        elif (0 <= r2 < self.maze_size and 0 <= c2 < self.maze_size and 
                              self.grid[r2, c2] == self.WALL):
                            # Look one step further
                            r3, c3 = r2 + dr, c2 + dc
                            
                            if (0 <= r3 < self.maze_size and 0 <= c3 < self.maze_size and 
                                (r3, c3) in accessible):
                                dist = 2  # Need to remove one wall
                                if dist < min_dist:
                                    min_dist = dist
                                    best_connection = ((r1, c1), (r3, c3))
                
                if best_connection:
                    # Create a path between the two regions
                    cell1, cell2 = best_connection
                    r1, c1 = cell1
                    r2, c2 = cell2
                    
                    # Calculate middle point (the wall to remove)
                    mid_r = (r1 + r2) // 2
                    mid_c = (c1 + c2) // 2
                    
                    # Remove the wall
                    self.grid[mid_r, mid_c] = self.EMPTY
                    
                    # Update sets - this region is now accessible
                    accessible.update(region)
                    inaccessible -= region
                else:
                    # If no connection found, just create a direct path to the agent
                    # (this should be rare if the maze is well-formed)
                    cell = next(iter(region))
                    self._create_somewhat_direct_path(cell, self.agent_position)
                    
                    # Update sets - this region is now accessible
                    accessible.update(region)
                    inaccessible -= region
        
        # Verify that the goal is accessible
        if self.goal_position not in accessible:
            # Create a direct path to the goal
            self._create_somewhat_direct_path(self.agent_position, self.goal_position)
        
        # Return whether all cells are now accessible
        return len(empty_cells - accessible) == 0
    
    def _create_multiple_paths(self):
        """Ensure there are multiple paths between key points in the maze."""
        # The recursive backtracking algorithm creates a perfect maze with exactly
        # one path between any two points. We've already added some loops with
        # _add_maze_loops(), but we can also ensure there's a good path to the goal.
        
        # Ensure there's at least one clear path to the goal
        if not self._is_maze_navigable():
            self._create_path_to_goal()
            
        # Create one more direct path from a random point to the goal
        # This ensures that even with all the twisty maze corridors, there's
        # at least one somewhat direct route
        
        # Find a random point in the first third of the maze
        r = random.randint(1, max(2, self.maze_size // 3))
        c = random.randint(1, max(2, self.maze_size // 3))
        random_point = (r, c)
        
        # Make sure this point is empty
        self.grid[random_point] = self.EMPTY
        
        # Create a more direct path to the goal
        self._create_somewhat_direct_path(random_point, self.goal_position)
    
    def _create_somewhat_direct_path(self, start, end):
        """Create a somewhat direct path from start to end, with some randomness."""
        current = start
        
        while current != end:
            r, c = current
            
            # 80% chance to move toward the goal, 20% chance for random movement
            if random.random() < 0.8:
                # Choose the direction that gets us closer to the goal
                if r < end[0]:
                    dr, dc = 1, 0  # Move down
                elif r > end[0]:
                    dr, dc = -1, 0  # Move up
                elif c < end[1]:
                    dr, dc = 0, 1  # Move right
                else:  # c > end[1]
                    dr, dc = 0, -1  # Move left
            else:
                # Choose a random direction
                dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            
            # Calculate new position
            new_r, new_c = r + dr, c + dc
            
            # Check bounds
            if 1 <= new_r < self.maze_size-1 and 1 <= new_c < self.maze_size-1:
                # Clear this cell
                self.grid[new_r, new_c] = self.EMPTY
                current = (new_r, new_c)
                
        # Make sure goal is properly marked
        self.grid[end] = self.GOAL
    
    def _create_path_to_goal(self):
        """Create a guaranteed path from agent to goal if one doesn't exist."""
        # Use a simple path-carving approach
        current = self.agent_position
        goal = self.goal_position
        
        while current != goal:
            # Move towards goal
            if current[0] < goal[0]:
                next_pos = (current[0] + 1, current[1])
            elif current[0] > goal[0]:
                next_pos = (current[0] - 1, current[1])
            elif current[1] < goal[1]:
                next_pos = (current[0], current[1] + 1)
            else:  # current[1] > goal[1]
                next_pos = (current[0], current[1] - 1)
            
            # Clear the path
            self.grid[next_pos] = self.EMPTY
            current = next_pos
        
        # Make sure goal is properly marked
        self.grid[goal] = self.GOAL
                    
        # Check if maze is still navigable
        if not self._is_maze_navigable():
            # If not navigable, remove some walls randomly until it becomes navigable
            wall_positions = [(r, c) for r in range(1, self.maze_size-1) 
                               for c in range(1, self.maze_size-1) 
                               if self.grid[r, c] == self.WALL and 
                               (r, c) != self.agent_position and 
                               (r, c) != self.goal_position]
            
            np.random.shuffle(wall_positions)
            
            for wall_pos in wall_positions:
                # Remove the wall
                self.grid[wall_pos] = self.EMPTY
                
                # Check if maze is now navigable
                if self._is_maze_navigable():
                    break
                
    def _create_simple_maze(self):
        """Create a simplified maze with thick walls and a clear path to the goal."""
        # Create a grid filled with walls
        self.grid = np.ones((self.maze_size, self.maze_size), dtype=np.uint8)
        
        # Make an "S" shaped path from top-left to bottom-right
        # This ensures a winding path like in the reference image
        
        # Top horizontal corridor
        for c in range(1, self.maze_size - 1):
            self.grid[1, c] = self.EMPTY
        
        # First vertical corridor
        for r in range(1, self.maze_size // 2):
            self.grid[r, self.maze_size - 2] = self.EMPTY
            
        # Middle horizontal corridor
        for c in range(2, self.maze_size - 1):
            self.grid[self.maze_size // 2, c] = self.EMPTY
            
        # Second vertical corridor
        for r in range(self.maze_size // 2, self.maze_size - 1):
            self.grid[r, 1] = self.EMPTY
            
        # Bottom horizontal corridor
        for c in range(1, self.maze_size - 1):
            self.grid[self.maze_size - 2, c] = self.EMPTY
        
        # Place agent in top-left
        self.agent_position = (1, 1)
        
        # Place goal in bottom-right
        self.goal_position = (self.maze_size-2, self.maze_size-2)
        self.grid[self.goal_position] = self.GOAL
        
        # Add a few random rooms or extra corridors to make it more interesting
        self._add_random_branches()
    
    def _add_random_branches(self):
        """Add random branches and rooms to the simple maze to make it more complex."""
        empty_cells = [(r, c) for r in range(1, self.maze_size-1) 
                      for c in range(1, self.maze_size-1) 
                      if self.grid[r, c] == self.EMPTY and
                      (r, c) != self.agent_position and
                      (r, c) != self.goal_position]
        
        # Add 3-5 branches
        num_branches = random.randint(3, min(5, len(empty_cells) // 3))
        
        for _ in range(num_branches):
            if not empty_cells:
                break
                
            # Select a random cell to start a branch
            start_cell = random.choice(empty_cells)
            empty_cells.remove(start_cell)
            
            # Determine branch length and direction
            branch_length = random.randint(2, 4)
            direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            
            # Create the branch
            current_r, current_c = start_cell
            for i in range(branch_length):
                new_r = current_r + direction[0]
                new_c = current_c + direction[1]
                
                # Check bounds and avoid crossing paths
                if (1 <= new_r < self.maze_size-1 and 
                    1 <= new_c < self.maze_size-1 and
                    self.grid[new_r, new_c] == self.WALL):
                    self.grid[new_r, new_c] = self.EMPTY
                    current_r, current_c = new_r, new_c
                else:
                    break
    
    def _is_maze_navigable(self):
        """Check if there is a path from agent position to goal."""
        return self._is_path_valid(self.agent_position, self.goal_position)
    
    def _is_path_valid(self, start_pos, end_pos):
        """Check if there is a valid path from start_pos to end_pos."""
        # Use BFS to find a path
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)
        
        while queue:
            current = queue.popleft()
            
            if current == end_pos:
                return True
            
            # Check all four adjacent cells
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (current[0] + dr, current[1] + dc)
                
                # Check if the position is valid and not a wall
                if (0 <= new_pos[0] < self.maze_size and 
                    0 <= new_pos[1] < self.maze_size and 
                    self.grid[new_pos] != self.WALL and
                    new_pos not in visited):
                    
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return False
    
    def _get_observation(self):
        """Convert the maze state to a multi-channel observation."""
        # Create a multi-channel representation
        obs = np.zeros((self.maze_size, self.maze_size, self.channels), dtype=np.float32)
        
        # Channel 0: Agent position
        agent_layer = np.zeros((self.maze_size, self.maze_size))
        agent_layer[self.agent_position] = 1.0
        obs[:, :, 0] = agent_layer
        
        # Channel 1: Walls
        wall_layer = (self.grid == self.WALL).astype(np.float32)
        obs[:, :, 1] = wall_layer
        
        # Channel 2: Goal
        goal_layer = (self.grid == self.GOAL).astype(np.float32)
        obs[:, :, 2] = goal_layer
        
        return obs
    
    def step(self, action):
        """Take a step in the environment with the given action."""
        # Increment step counter
        self.steps_done += 1
        
        # Current position
        r, c = self.agent_position
        
        # Determine next position based on action
        # 0=up, 1=right, 2=down, 3=left
        if action == 0:  # Up
            next_pos = (r - 1, c)
        elif action == 1:  # Right
            next_pos = (r, c + 1)
        elif action == 2:  # Down
            next_pos = (r + 1, c)
        elif action == 3:  # Left
            next_pos = (r, c - 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Check if next position is valid (not a wall and within bounds)
        if (0 <= next_pos[0] < self.maze_size and 
            0 <= next_pos[1] < self.maze_size and
            self.grid[next_pos] != self.WALL):
            self.agent_position = next_pos
        
        # Check if reached goal
        done = tuple(self.agent_position) == tuple(self.goal_position)
        
        # Initialize reward with step penalty
        reward = -0.01  # Small penalty for each step
        
        # Add goal reward if goal reached
        if done:
            reward += 10.0  # Reward for reaching goal
        
        # Add exploration bonus for visiting new states
        is_new_state = self.agent_position not in self.visited_positions
        if is_new_state:
            reward += 0.05  # Exploration bonus
            self.visited_positions.add(self.agent_position)
        else:
            # Still track visited positions even if no bonus
            self.visited_positions.add(self.agent_position)
        
        # Info dict for additional data
        info = {
            'step_count': self.steps_done,
            'agent_position': self.agent_position,
            'visited_positions': len(self.visited_positions),
            'success': done  # Flag for success metrics
        }
        
        # Check for episode termination due to max steps
        truncated = self.steps_done >= self.max_steps
        
        # Return observation, reward, done, truncated, info (gym v26+ format)
        return self._get_observation(), reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment to a new episode."""
        # Set the random seed if provided and not using fixed seed
        if seed is not None and not self.use_fixed_seed:
            self.seed(seed)
        
        # Generate a new maze if not using fixed layouts
        if not self.use_fixed_layout:
            self.generate_maze()
        else:
            # Just reset agent position
            self.agent_position = (1, 1)  # Top-left corner, inside the outer wall
        
        # Reset step counter and visited positions
        self.steps_done = 0
        self.visited_positions = {self.agent_position}
        
        # Return initial observation
        obs = self._get_observation()
        return obs, {}
    
    def render(self, mode='human'):
        """Render the current state of the environment."""
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
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
            
            self.ax.clear()
            self.ax.imshow(maze_img)
            
            # Add gridlines
            self.ax.set_xticks(np.arange(-0.5, self.maze_size, 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.maze_size, 1), minor=True)
            self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            self.ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            
            # Create legend elements
            legend_elements = [
                mpatches.Patch(color=self.COLORS[self.EMPTY], label='Empty'),
                mpatches.Patch(color=self.COLORS[self.WALL], label='Wall'),
                mpatches.Patch(color=self.COLORS[self.GOAL], label='Goal'),
                mpatches.Patch(color=self.COLORS[self.AGENT], label='Agent')
            ]
            
            # Add legend
            self.ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add step counter
            self.ax.set_title(f"Steps: {self.steps_done}/{self.max_steps}")
            
            plt.tight_layout()
            plt.pause(0.1)
            return self.fig
            
        elif mode == 'rgb_array':
            return (maze_img * 255).astype(np.uint8)
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def get_coverage_stats(self):
        """Get statistics about environment coverage."""
        # Count total navigable cells (not walls)
        total_navigable = sum(1 for r in range(self.maze_size) 
                             for c in range(self.maze_size) 
                             if self.grid[r, c] != self.WALL)
        
        # Count visited cells
        visited = len(self.visited_positions)
        
        # Calculate coverage percentage
        coverage = visited / total_navigable if total_navigable > 0 else 0
        
        return {
            'coverage': coverage,
            'visited': visited,
            'total_navigable': total_navigable
        }
