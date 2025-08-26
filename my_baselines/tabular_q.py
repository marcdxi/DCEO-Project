"""
Tabular Q-learning agent for discrete state spaces.
Especially suitable for testing with the key maze environment.
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class TabularQAgent:
    """
    Tabular Q-learning agent for discrete state spaces.
    Uses a simple dictionary to store Q-values.
    """
    
    def __init__(self, 
                 num_actions, 
                 learning_rate=0.1,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 count_bonus_weight=0.1):
        """
        Initialize the Q-learning agent.
        
        Args:
            num_actions: Number of possible actions
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            count_bonus_weight: Weight for count-based exploration bonus
        """
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.count_bonus_weight = count_bonus_weight
        
        # Initialize Q-table and visit counts
        self.q_table = {}
        self.state_visits = {}
        self.steps = 0
        
        # Episode tracking
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _get_state_key(self, state):
        """
        Convert state to a hashable representation for the Q-table.
        For KeyDoorMazeEnv, we'll use the agent position, key status, and door status.
        """
        if hasattr(state, 'agent_position') and hasattr(state, 'inventory') and hasattr(state, 'door_status'):
            # If state is the environment itself (for debugging)
            agent_pos = state.agent_position
            keys = tuple(state.inventory)
            doors = tuple(state.door_status)
            return (agent_pos, keys, doors)
        
        # For wrapped environment states (numpy arrays)
        if isinstance(state, np.ndarray):
            # If state is a 3D array (image-like representation)
            if len(state.shape) == 3:
                # Find agent position (marked in channel 0 in our wrapper)
                # For KeyMazeWrapper, agent position is highlighted in channel 0
                if state.shape[0] == 5:  # KeyMaze has 5 channels
                    # Find the agent position (marked as 0.5 in our wrapper)
                    agent_pos = np.unravel_index(np.argmax(state[0] == 0.5), state[0].shape)
                    
                    # Check if keys collected (channel 3)
                    keys_collected = np.max(state[3]) < 0.5  # If no value >0.5 in key channel, key is collected
                    
                    # Check if doors open (channel 4)
                    doors_open = np.max(state[4]) < 0.5  # If no value >0.5 in door channel, door is open
                    
                    return (agent_pos, (keys_collected,), (doors_open,))
            
        # Fallback: just use the string representation of the state
        return str(state)
    
    def select_action(self, state, explore=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to use exploration
            
        Returns:
            Selected action
        """
        state_key = self._get_state_key(state)
        
        # Initialize state entry if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        
        # Update visit count for exploration bonus
        self.state_visits[state_key] = self.state_visits.get(state_key, 0) + 1
        
        # Track state visits without debugging prints
            
        if explore and np.random.random() < self.epsilon:
            # Random action
            action = np.random.randint(self.num_actions)
            return action
        else:
            # Greedy action
            action = np.argmax(self.q_table[state_key])
            return action
    
    def get_intrinsic_reward(self, state):
        """
        Calculate intrinsic reward based on state visit counts.
        
        Args:
            state: Current state
            
        Returns:
            Intrinsic reward value
        """
        state_key = self._get_state_key(state)
        visits = self.state_visits.get(state_key, 0)
        return self.count_bonus_weight / np.sqrt(visits + 1)
    
    def step(self, state, action, reward, next_state, done, episode):
        """
        Update Q-table based on the observed transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            episode: Current episode number
        """
        self.steps += 1
        
        # Update visit count for state-action pair
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_actions)
        
        # Calculate intrinsic reward
        intrinsic_reward = self.get_intrinsic_reward(state)
        total_reward = reward + intrinsic_reward
        
        # Q-learning update
        if not done:
            target = total_reward + self.gamma * np.max(self.q_table[next_state_key])
        else:
            target = total_reward
            
        # Update Q-value
        self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Track episode reward
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
    def save(self, path):
        """
        Save the agent's Q-table and other parameters.
        
        Args:
            path: Path to save the agent
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            'q_table': self.q_table,
            'state_visits': self.state_visits,
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episode_rewards': self.episode_rewards
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
            
    def load(self, path):
        """
        Load the agent's Q-table and other parameters.
        
        Args:
            path: Path to load the agent from
        """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
            
        self.q_table = state_dict['q_table']
        self.state_visits = state_dict['state_visits']
        self.epsilon = state_dict['epsilon']
        self.steps = state_dict['steps']
        self.episode_rewards = state_dict['episode_rewards']
        
    def plot_rewards(self, output_path=None):
        """
        Plot the agent's episode rewards.
        
        Args:
            output_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        
        if output_path:
            plt.savefig(output_path)
        plt.show()
        
    def plot_state_visits(self, maze_shape=(12, 12), output_path=None):
        """
        Plot the state visitation heatmap.
        
        Args:
            maze_shape: Shape of the maze
            output_path: Path to save the plot
        """
        visits_map = np.zeros(maze_shape)
        
        for state_key, visits in self.state_visits.items():
            if isinstance(state_key, tuple) and isinstance(state_key[0], tuple) and len(state_key[0]) == 2:
                i, j = state_key[0]
                if 0 <= i < maze_shape[0] and 0 <= j < maze_shape[1]:
                    visits_map[i, j] = visits
        
        plt.figure(figsize=(10, 8))
        plt.imshow(visits_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Visit Count')
        plt.title('State Visitation Heatmap')
        
        if output_path:
            plt.savefig(output_path)
        plt.show()
