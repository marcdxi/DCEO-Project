"""
Key-Door Maze Environment for Reinforcement Learning.

This environment implements a maze with keys, doors, and a goal.
The agent must collect keys to open doors and reach the goal.
"""

import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import random
from collections import deque

class KeyDoorMazeEnv(gym.Env):
    """
    A Key-Door Maze environment compatible with Rainbow DCEO agents.
    
    The maze contains walls, keys, doors, and a goal. The agent must
    collect keys to open corresponding doors, and reach the goal.
    
    Actions:
    - 0: Move up
    - 1: Move right
    - 2: Move down
    - 3: Move left
    
    Rewards:
    - Collecting a key: +0.5
    - Opening a door: +1.0
    - Reaching the goal: +10.0
    - Every step: -0.01 (small penalty to encourage efficient paths)
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    # Cell types
    EMPTY = 0
    WALL = 1
    GOAL = 2
    KEY = 3
    DOOR = 4
    AGENT = 5
    
    # Colors for rendering
    COLORS = {
        EMPTY: [1.0, 1.0, 1.0],  # White
        WALL: [0.0, 0.0, 0.0],   # Black
        GOAL: [0.0, 1.0, 0.0],   # Green
        KEY: [1.0, 1.0, 0.0],    # Yellow
        DOOR: [0.7, 0.3, 0.0],   # Brown
        AGENT: [1.0, 0.0, 0.0]   # Red
    }
    
    def __init__(self, maze_size=10, num_keys=1, max_steps=100, use_fixed_layout=False, use_fixed_seed=False, fixed_seed=42):
        """Initialize the Key-Door Maze environment.
        
        Args:
            maze_size: Size of the maze grid (maze_size x maze_size)
            num_keys: Number of keys and doors in the maze
            max_steps: Maximum steps per episode
            use_fixed_seed: Whether to use a fixed seed for maze generation
            fixed_seed: The fixed seed to use
            use_fixed_layout: Whether to reuse maze layout across episodes
        """
        super(KeyDoorMazeEnv, self).__init__()
        
        self.maze_size = maze_size
        self.num_keys = max(1, num_keys)  # Ensure at least one key to make the environment challenging
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
        # Channel 3: Keys
        # Channel 4: Doors
        self.channels = 5
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
        self.inventory = [False] * num_keys  # Track collected keys
        self.door_status = [False] * num_keys  # Track opened doors
        
    def seed(self, seed=None):
        """Set the random seed for this environment."""
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
        
    def generate_maze(self):
        """Generate a new maze with door next to goal and goal inaccessible without the door."""
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
            self._add_enhanced_internal_walls()
            
            # Create wall enclosure around goal with one opening
            self._create_goal_enclosure()
            
            # Place door at the entrance to goal enclosure
            self.door_positions = [self._place_door_at_goal_entrance()]
            
            # Verify the maze with door is navigable
            if not self._is_maze_navigable_without_door():
                print(f"Attempt {attempt+1}: Maze is not navigable, regenerating...")
                continue
            
            # Add keys in positions reachable from start
            self.key_positions = []
            for i in range(self.num_keys):
                key_pos = self._find_reachable_empty_position(from_pos=self.agent_position)
                if key_pos:
                    self.key_positions.append(key_pos)
                    self.grid[key_pos] = self.KEY
                else:
                    # If we couldn't find a valid position, try again
                    print(f"Attempt {attempt+1}: Could not place key {i+1}, regenerating maze...")
                    break
            
            # If we placed all keys successfully, validate maze
            if len(self.key_positions) == self.num_keys:
                # Reset inventory and step counter
                self.inventory = [False] * self.num_keys
                self.door_status = [False] * self.num_keys
                self.steps_done = 0
                self.visited_positions = {self.agent_position}
                
                # Check that maze is solvable (can get key and reach goal)
                if self._validate_enhanced_maze():
                    print(f"Generated valid maze after {attempt+1} attempts")
                    break
                else:
                    print(f"Attempt {attempt+1}: Generated maze is not solvable, regenerating...")
        
        # If we exhausted all attempts, create a simple maze with required properties
        if attempt == max_attempts - 1:
            print("Exhausted maximum attempts. Creating simplified maze...")
            self._create_enhanced_simple_maze()
        
    def _add_outer_walls(self):
        """Add walls around the perimeter of the maze."""
        # Top and bottom walls
        self.grid[0, :] = self.WALL
        self.grid[self.maze_size-1, :] = self.WALL
        
        # Left and right walls
        self.grid[:, 0] = self.WALL
        self.grid[:, self.maze_size-1] = self.WALL
        
    def _add_internal_walls(self):
        """Add random internal walls to make the maze more interesting."""
        # Add some horizontal and vertical walls with gaps
        # Reduce the number of walls for better navigation
        num_walls = max(1, self.maze_size // 4)  # Fewer walls for better navigation
        
        for _ in range(num_walls):
            # Add horizontal wall segment
            row = np.random.randint(2, self.maze_size-2)
            col_start = np.random.randint(1, self.maze_size-4)
            length = np.random.randint(2, min(4, self.maze_size-col_start-1))  # Shorter walls
            
            # Leave multiple gaps in the wall to ensure paths exist
            num_gaps = max(1, length // 2)
            gap_positions = set(np.random.choice(range(col_start, col_start+length), num_gaps, replace=False))
            
            for col in range(col_start, col_start+length):
                if col not in gap_positions:
                    self.grid[row, col] = self.WALL
            
            # Add vertical wall segment - similar improvements as horizontal
            col = np.random.randint(2, self.maze_size-2)
            row_start = np.random.randint(1, self.maze_size-4)
            length = np.random.randint(2, min(4, self.maze_size-row_start-1))  # Shorter walls
            
            # Leave multiple gaps in the wall to ensure paths exist
            num_gaps = max(1, length // 2)
            gap_positions = set(np.random.choice(range(row_start, row_start+length), num_gaps, replace=False))
            
            for row in range(row_start, row_start+length):
                if row not in gap_positions:
                    self.grid[row, col] = self.WALL
                    
    def _add_enhanced_internal_walls(self):
        """Add more internal walls while ensuring the maze remains navigable."""
        # Add more walls than the original method
        num_walls = max(2, self.maze_size // 3)  # More walls for increased complexity
        
        for _ in range(num_walls):
            # Add horizontal wall segment
            row = np.random.randint(2, self.maze_size-2)
            col_start = np.random.randint(1, self.maze_size-4)
            length = np.random.randint(3, min(5, self.maze_size-col_start-1))  # Longer walls
            
            # Leave fewer gaps in the wall to make navigation more challenging
            num_gaps = max(1, length // 3)  # Fewer gaps
            gap_positions = set(np.random.choice(range(col_start, col_start+length), num_gaps, replace=False))
            
            for col in range(col_start, col_start+length):
                if col not in gap_positions:
                    self.grid[row, col] = self.WALL
            
            # Add vertical wall segment
            col = np.random.randint(2, self.maze_size-2)
            row_start = np.random.randint(1, self.maze_size-4)
            length = np.random.randint(3, min(5, self.maze_size-row_start-1))  # Longer walls
            
            # Leave fewer gaps
            num_gaps = max(1, length // 3)
            gap_positions = set(np.random.choice(range(row_start, row_start+length), num_gaps, replace=False))
            
            for row in range(row_start, row_start+length):
                if row not in gap_positions:
                    self.grid[row, col] = self.WALL
                    
        # Check if maze is still navigable without doors after adding walls
        if not self._is_maze_navigable_without_door():
            # If not navigable, remove some walls randomly until it becomes navigable
            wall_positions = [(r, c) for r in range(1, self.maze_size-1) 
                              for c in range(1, self.maze_size-1) 
                              if self.grid[r, c] == self.WALL and (r, c) != self.goal_position]
            
            while wall_positions and not self._is_maze_navigable_without_door():
                # Remove a random wall
                wall_pos = random.choice(wall_positions)
                self.grid[wall_pos] = self.EMPTY
                wall_positions.remove(wall_pos)
                
    def _create_goal_enclosure(self):
        """Create a wall enclosure around the goal with one opening."""
        goal_row, goal_col = self.goal_position
        
        # Determine direction for door (randomly select from valid directions)
        valid_directions = []
        
        # Check each direction (up, right, down, left)
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            check_r, check_c = goal_row + dr, goal_col + dc
            
            # Check if this position is inside the maze and not on the outer wall
            if (1 <= check_r < self.maze_size-1 and 
                1 <= check_c < self.maze_size-1):
                valid_directions.append((dr, dc))
        
        # If no valid directions, default to up
        if not valid_directions:
            direction = (-1, 0)  # Up
        else:
            direction = random.choice(valid_directions)
        
        # Create the door position
        door_row = goal_row + direction[0]
        door_col = goal_col + direction[1]
        self.goal_entrance = (door_row, door_col)
        
        # Build walls around the goal except at the door position
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            wall_r, wall_c = goal_row + dr, goal_col + dc
            
            # Ensure we're not placing walls outside the maze
            if (1 <= wall_r < self.maze_size-1 and 
                1 <= wall_c < self.maze_size-1 and 
                (wall_r, wall_c) != self.goal_entrance):
                self.grid[wall_r, wall_c] = self.WALL
                
    def _place_door_at_goal_entrance(self):
        """Place a door at the entrance to the goal enclosure."""
        door_pos = self.goal_entrance
        self.grid[door_pos] = self.DOOR
        return door_pos
        
    def _is_maze_navigable_without_door(self):
        """Check if the maze is navigable ignoring the door."""
        # Make a temporary copy of the grid
        temp_grid = self.grid.copy()
        
        # Temporarily remove all doors
        for r in range(self.maze_size):
            for c in range(self.maze_size):
                if temp_grid[r, c] == self.DOOR:
                    temp_grid[r, c] = self.EMPTY
        
        # Save original grid and use temp grid
        original_grid = self.grid.copy()
        self.grid = temp_grid
        
        # Check if agent can reach all empty spaces
        accessible = self._get_accessible_positions(self.agent_position)
        
        # Check if all empty cells are accessible
        all_accessible = True
        for r in range(1, self.maze_size-1):
            for c in range(1, self.maze_size-1):
                if (temp_grid[r, c] == self.EMPTY or temp_grid[r, c] == self.KEY or 
                    temp_grid[r, c] == self.GOAL) and (r, c) not in accessible:
                    all_accessible = False
                    break
                    
        # Restore original grid
        self.grid = original_grid
        
        return all_accessible
        
    def _get_accessible_positions(self, start_pos):
        """Get all positions accessible from start_pos."""
        accessible = set()
        queue = deque([start_pos])
        accessible.add(start_pos)
        
        while queue:
            current = queue.popleft()
            
            # Check all four adjacent cells
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (current[0] + dr, current[1] + dc)
                
                # Check if position is valid, not a wall or door, and not visited
                if (0 <= new_pos[0] < self.maze_size and 
                    0 <= new_pos[1] < self.maze_size and 
                    self.grid[new_pos] != self.WALL and
                    self.grid[new_pos] != self.DOOR and
                    new_pos not in accessible):
                    
                    accessible.add(new_pos)
                    queue.append(new_pos)
                    
        return accessible
        
    def _validate_enhanced_maze(self):
        """Validate that the maze is solvable with the special door-goal arrangement."""
        # Check if agent can reach the key
        for key_pos in self.key_positions:
            if not self._is_path_valid(self.agent_position, key_pos):
                return False
                
        # Create a temp grid with the door removed (as if it's open)
        temp_grid = self.grid.copy()
        for door_pos in self.door_positions:
            temp_grid[door_pos] = self.EMPTY
            
        # Save and switch to temp grid
        original_grid = self.grid.copy()
        self.grid = temp_grid
        
        # Check if agent can reach goal with door open
        goal_accessible = self._is_path_valid(self.agent_position, self.goal_position)
        
        # Restore original grid
        self.grid = original_grid
        
        return goal_accessible
        
    def _create_enhanced_simple_maze(self):
        """Create a simplified maze with the door next to goal requirement."""
        # Reset the grid
        self.grid = np.zeros((self.maze_size, self.maze_size), dtype=np.uint8)
        
        # Add outer walls
        self._add_outer_walls()
        
        # Place agent in top-left
        self.agent_position = (1, 1)
        
        # Place goal in bottom-right
        self.goal_position = (self.maze_size-2, self.maze_size-2)
        self.grid[self.goal_position] = self.GOAL
        
        # Create goal enclosure with one opening
        self._create_goal_enclosure()
        
        # Place door at goal entrance
        self.door_positions = [self._place_door_at_goal_entrance()]
        
        # Add a simple path with some walls
        for r in range(2, self.maze_size-2, 2):
            for c in range(1, self.maze_size-1):
                if c != 2:  # Leave a gap
                    self.grid[r, c] = self.WALL
                    
        # Place a key in a position reachable from start
        self.key_positions = []
        key_row, key_col = 3, 3
        while len(self.key_positions) < self.num_keys:
            if self.grid[key_row, key_col] == self.EMPTY:
                self.key_positions.append((key_row, key_col))
                self.grid[key_row, key_col] = self.KEY
                key_row += 2
                key_col += 2
            else:
                key_row += 1
                key_col += 1
                
        # Reset inventory and step counter
        self.inventory = [False] * self.num_keys
        self.door_status = [False] * self.num_keys
        self.steps_done = 0
        self.visited_positions = {self.agent_position}
    
    def _validate_maze_solvable(self):
        """Check if the maze is solvable (agent can collect all keys, open all doors, and reach goal)."""
        # Check if agent can reach at least one key
        can_reach_any_key = False
        for key_pos in self.key_positions:
            if self._is_path_valid(self.agent_position, key_pos):
                can_reach_any_key = True
                break
        
        if not can_reach_any_key:
            return False
        
        # Simulate collecting all keys (assume agent can get all keys)
        # Check if with all keys, agent can open all doors and reach goal
        temp_grid = self.grid.copy()
        
        # Temporarily remove doors (as if they're all open)
        for door_pos in self.door_positions:
            temp_grid[door_pos] = self.EMPTY
        
        # Save current grid and restore after check
        original_grid = self.grid.copy()
        self.grid = temp_grid
        
        # Check if agent can reach goal with all doors open
        path_to_goal_exists = self._is_path_valid(self.agent_position, self.goal_position)
        
        # Restore original grid
        self.grid = original_grid
        
        return path_to_goal_exists
    
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
                
                # Check if the position is valid and not a wall or door
                if (0 <= new_pos[0] < self.maze_size and 
                    0 <= new_pos[1] < self.maze_size and 
                    self.grid[new_pos] != self.WALL and
                    self.grid[new_pos] != self.DOOR and
                    new_pos not in visited):
                    
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return False
    
    def _find_valid_door_positions(self):
        """Find valid positions for doors that don't completely block paths."""
        valid_positions = []
        
        # Try positions in the middle section of the maze
        mid_row_start = self.maze_size // 3
        mid_row_end = 2 * self.maze_size // 3
        mid_col_start = self.maze_size // 3
        mid_col_end = 2 * self.maze_size // 3
        
        for r in range(mid_row_start, mid_row_end):
            for c in range(mid_col_start, mid_col_end):
                if self.grid[r, c] == self.EMPTY:
                    # Temporarily place a door and check if path still exists
                    self.grid[r, c] = self.DOOR
                    
                    # Verify we don't block all paths from agent to goal
                    # We need at least one other potential path
                    if self._count_potential_paths(self.agent_position, self.goal_position) > 0:
                        valid_positions.append((r, c))
                    
                    # Remove the temporary door
                    self.grid[r, c] = self.EMPTY
        
        return valid_positions
    
    def _count_potential_paths(self, start_pos, end_pos):
        """Count number of potential paths from start to end (ignoring doors)."""
        # Make a copy of the grid with doors treated as empty
        temp_grid = self.grid.copy()
        for r in range(self.maze_size):
            for c in range(self.maze_size):
                if temp_grid[r, c] == self.DOOR:
                    temp_grid[r, c] = self.EMPTY
        
        # Save and restore original grid
        original_grid = self.grid.copy()
        self.grid = temp_grid
        
        # Use BFS to count paths
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)
        path_count = 0
        
        while queue:
            current = queue.popleft()
            
            if current == end_pos:
                path_count += 1
                continue
            
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
        
        # Restore original grid
        self.grid = original_grid
        
        return path_count
    
    def _create_valid_door_position(self):
        """Create a valid door position by clearing space if needed."""
        # Find a position that doesn't completely block the maze
        for r in range(1, self.maze_size-1):
            for c in range(1, self.maze_size-1):
                if self.grid[r, c] == self.EMPTY:
                    # Temporarily place door
                    self.grid[r, c] = self.DOOR
                    
                    # Check if there's still a path from agent to goal (ignoring other doors)
                    has_path = self._count_potential_paths(self.agent_position, self.goal_position) > 0
                    
                    # Remove temporary door
                    self.grid[r, c] = self.EMPTY
                    
                    if has_path:
                        return (r, c)
        
        return None
    
    def _find_reachable_empty_position(self, from_pos):
        """Find an empty position that is reachable from the given position."""
        # Get all reachable positions from from_pos
        visited = set()
        queue = deque([from_pos])
        visited.add(from_pos)
        reachable = []
        
        while queue:
            current = queue.popleft()
            
            # If this is an empty position, add it to reachable
            if self.grid[current] == self.EMPTY:
                reachable.append(current)
            
            # Check all four adjacent cells
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (current[0] + dr, current[1] + dc)
                
                # Check if the position is valid and not a wall or door
                if (0 <= new_pos[0] < self.maze_size and 
                    0 <= new_pos[1] < self.maze_size and 
                    self.grid[new_pos] != self.WALL and
                    self.grid[new_pos] != self.DOOR and
                    new_pos not in visited):
                    
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        # Return a random reachable empty position, or None if none found
        if reachable:
            return random.choice(reachable)
        return None
    
    def _create_simple_maze(self):
        """Create a very simple, guaranteed solvable maze."""
        # Reset the grid to empty
        self.grid = np.zeros((self.maze_size, self.maze_size), dtype=np.uint8)
        
        # Add only outer walls
        self._add_outer_walls()
        
        # Place agent in top-left
        self.agent_position = (1, 1)
        
        # Place goal in bottom-right
        self.goal_position = (self.maze_size-2, self.maze_size-2)
        self.grid[self.goal_position] = self.GOAL
        
        # Add a simple path with doors and keys
        self.door_positions = []
        self.key_positions = []
        
        # Place keys and doors in a solvable configuration
        third = self.maze_size // 3
        two_thirds = 2 * self.maze_size // 3
        
        # First key near agent
        key_pos = (1, third)
        self.key_positions.append(key_pos)
        self.grid[key_pos] = self.KEY
        
        # First door after first key
        door_pos = (1, two_thirds)
        self.door_positions.append(door_pos)
        self.grid[door_pos] = self.DOOR
        
        # Additional keys and doors
        for i in range(1, self.num_keys):
            # Place key at position i
            key_pos = (i+1, third)
            self.key_positions.append(key_pos)
            self.grid[key_pos] = self.KEY
            
            # Place door at position i
            door_pos = (i+1, two_thirds)
            self.door_positions.append(door_pos)
            self.grid[door_pos] = self.DOOR
        
        # Reset inventory and step counter
        self.inventory = [False] * self.num_keys
        self.door_status = [False] * self.num_keys
        self.steps_done = 0
        self.visited_positions = {self.agent_position}
        
    def reset(self, seed=None, options=None):
        """Reset the environment to a new episode.
        
        Args:
            seed: Random seed to use for maze generation (if not fixed)
            options: Additional options (not used currently)
            
        Returns:
            obs: Initial observation
            info: Empty info dictionary
        """
        # Set the random seed if provided and not using fixed seed
        if seed is not None and not self.use_fixed_seed:
            self.seed(seed)
            
        # Generate a new maze if not using fixed layouts, otherwise reuse the same layout
        if not self.use_fixed_layout:
            self.generate_maze()
        else:
            # Just reset agent position, inventory, keys, doors
            self.agent_position = (1, 1)  # Top-left corner, inside the outer wall
            
            # Reset keys and doors to be visible again
            for pos in self.key_positions:
                self.grid[pos] = self.KEY
                
            for pos in self.door_positions:
                self.grid[pos] = self.DOOR
        
        # Reset inventory and step counter
        self.inventory = [False] * self.num_keys
        self.door_status = [False] * self.num_keys
        self.steps_done = 0
        self.visited_positions = {self.agent_position}
        
        # Return initial observation
        obs = self._get_observation()
        return obs, {}
        
    def step(self, action):
        """Take a step in the environment with the given action."""
        # Increment step counter
        self.steps_done += 1
        
        # Default values - step penalty to encourage efficiency
        reward = -0.01  # Small negative reward per step as specified
        done = False
        truncated = False
        
        # Provide detailed info dictionary for debugging
        info = {
            'keys_collected': sum(self.inventory),
            'doors_opened': sum(self.door_status),
            'goal_reached': False,
            'key_collected': False,
            'door_opened': False,
            'reached_goal': False,
            'agent_position': self.agent_position,
            'steps_done': self.steps_done,
            'inventory': self.inventory.copy(),
            'door_status': self.door_status.copy(),
            'total_keys': len(self.key_positions),
            'total_doors': len(self.door_positions)
        }
        
        # Check if max steps reached
        if self.steps_done >= self.max_steps:
            truncated = True
        
        # Move the agent based on action
        new_position = self._get_new_position(action)
        
        # Check if the move is valid
        if self._is_valid_move(new_position):
            # Update agent position
            self.agent_position = new_position
            self.visited_positions.add(new_position)
            
            # Check cell type at new position
            cell_type = self.grid[new_position]
            
            if cell_type == self.GOAL:
                # Agent reached the goal - check if they have all required keys AND have opened all doors
                if all(self.inventory) and all(self.door_status):
                    # Agent has all keys and opened all doors - reaching the goal
                    reward += 1.0  # Goal completion reward as specified
                    done = True
                    info['goal_reached'] = True
                elif not all(self.inventory):
                    # Agent doesn't have all keys - cannot reach goal yet
                    # Move agent back to previous position
                    self.agent_position = self._get_previous_position(action)
                    # Add a small negative reward to discourage attempting the goal without keys
                    reward -= 0.1
                    info['blocked_at_goal'] = True
                    print(f"Cannot reach goal: Missing {sum(1 for k in self.inventory if not k)}/{len(self.inventory)} keys")
                elif not all(self.door_status):
                    # Agent has all keys but hasn't opened all doors
                    # Move agent back to previous position
                    self.agent_position = self._get_previous_position(action)
                    # Add a small negative reward
                    reward -= 0.1
                    info['blocked_at_goal'] = True
                    print(f"Cannot reach goal: {sum(1 for d in self.door_status if not d)}/{len(self.door_status)} doors remain closed")
                
            elif cell_type == self.KEY:
                # Agent found a key - increase reward significantly
                key_idx = self.key_positions.index(new_position)
                if not self.inventory[key_idx]:
                    self.inventory[key_idx] = True
                    reward += 0.3  # Key collection reward as specified
                    info['key_collected'] = True
                    
                # Clear the key from the grid
                self.grid[new_position] = self.EMPTY
                
            elif cell_type == self.DOOR:
                # Agent at a door - check if they have the key
                door_idx = self.door_positions.index(new_position)
                if self.inventory[door_idx] and not self.door_status[door_idx]:
                    # Open the door - increased reward
                    self.door_status[door_idx] = True
                    self.grid[new_position] = self.EMPTY
                    reward += 0.3  # Door opening reward as specified
                    info['door_opened'] = True
                elif not self.door_status[door_idx]:
                    # Door is locked and agent doesn't have the key
                    # Move agent back to previous position
                    self.agent_position = self._get_previous_position(action)
        else:
            # Invalid move (into wall or locked door) - do nothing
            pass
            
        # Middle-ground reward structure - no proximity-based shaping rewards
        # This allows more exploration while still providing key signals
        
        # Track distances but don't use them for shaping rewards
        agent_row, agent_col = self.agent_position
        
        # Just record distances for tracking purposes
        uncollected_keys = [pos for i, pos in enumerate(self.key_positions) if not self.inventory[i]]
        if uncollected_keys:
            # Find distance to closest uncollected key
            key_distances = [abs(pos[0] - agent_row) + abs(pos[1] - agent_col) for pos in uncollected_keys]
            min_key_dist = min(key_distances) if key_distances else self.maze_size * 2
            self.prev_key_dist = min_key_dist
        
        # Also track goal distance but don't reward based on it
        if all(self.inventory) and not done:
            # Distance to goal
            goal_dist = abs(self.goal_position[0] - agent_row) + abs(self.goal_position[1] - agent_col)
            self.prev_goal_dist = goal_dist
            
        # Return observation, reward, done, info
        obs = self._get_observation()
        return obs, reward, done, truncated, info
        
    def _get_new_position(self, action):
        """Get the new position after taking an action."""
        row, col = self.agent_position
        
        if action == 0:  # UP
            return (max(0, row-1), col)
        elif action == 1:  # RIGHT
            return (row, min(self.maze_size-1, col+1))
        elif action == 2:  # DOWN
            return (min(self.maze_size-1, row+1), col)
        elif action == 3:  # LEFT
            return (row, max(0, col-1))
        else:
            # Invalid action - stay in place
            return self.agent_position
            
    def _get_previous_position(self, action):
        """Get the previous position (undo an action)."""
        # Opposite actions: UP(0)<->DOWN(2), RIGHT(1)<->LEFT(3)
        opposite_action = (action + 2) % 4
        return self._get_new_position(opposite_action)
        
    def _is_valid_move(self, position):
        """Check if a move to the given position is valid."""
        cell_type = self.grid[position]
        
        if cell_type == self.WALL:
            # Can't move into walls
            return False
            
        if cell_type == self.DOOR:
            # Check if the door is open or if agent has the key
            door_idx = self.door_positions.index(position)
            if not self.door_status[door_idx] and not self.inventory[door_idx]:
                # Door is locked and agent doesn't have the key
                return False
                
        # All other cells are valid moves
        return True
        
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
        
        # Channel 3: Keys (visible keys that haven't been collected)
        key_layer = (self.grid == self.KEY).astype(np.float32)
        obs[:, :, 3] = key_layer
        
        # Channel 4: Doors (visible doors that haven't been opened)
        door_layer = (self.grid == self.DOOR).astype(np.float32)
        obs[:, :, 4] = door_layer
        
        return obs
        
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
                    
        # Display using matplotlib - non-blocking version
        if mode == 'human':
            # Use a single figure for continuous updates
            if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
                self.fig, self.ax = plt.subplots(figsize=(6, 6))
                self.legend_drawn = False
            
            # Clear previous plot and draw new one
            self.ax.clear()
            self.ax.imshow(maze_img)
            
            # Create legend (only once to avoid cluttering)
            if not hasattr(self, 'legend_drawn') or not self.legend_drawn:
                legend_elements = [
                    mpatches.Patch(color=self.COLORS[self.EMPTY], label='Empty'),
                    mpatches.Patch(color=self.COLORS[self.WALL], label='Wall'),
                    mpatches.Patch(color=self.COLORS[self.GOAL], label='Goal'),
                    mpatches.Patch(color=self.COLORS[self.KEY], label='Key'),
                    mpatches.Patch(color=self.COLORS[self.DOOR], label='Door'),
                    mpatches.Patch(color=self.COLORS[self.AGENT], label='Agent')
                ]
                self.ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
                self.legend_drawn = True
            
            # Display information about keys and doors
            keys_text = "Keys: " + "".join(["🔑" if k else "❌" for k in self.inventory])
            doors_text = "Doors: " + "".join(["🚪" if d else "🔒" for d in self.door_status])
            steps_text = f"Steps: {self.steps_done}/{self.max_steps}"
            
            self.ax.set_title(f"{keys_text}  |  {doors_text}  |  {steps_text}")
            plt.tight_layout()
            
            # Use draw and pause for non-blocking updates (like ComplexMazeEnv)
            plt.draw()
            plt.pause(0.01)  # Small pause to allow the plot to update
            
        elif mode == 'rgb_array':
            return (maze_img * 255).astype(np.uint8)
            
    def get_coverage_stats(self):
        """Return statistics about the agent's coverage of the maze."""
        visited_count = len(self.visited_positions)
        total_navigable = np.sum(self.grid != self.WALL)
        coverage = visited_count / total_navigable
        
        return {
            'visited_count': visited_count,
            'total_navigable': total_navigable,
            'coverage': coverage
        }
