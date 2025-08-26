"""
Simple test script for the Key-Door Maze environment with a small network.
This uses a smaller convolutional kernel appropriate for a small maze.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random

# Direct import for the key-maze environment
import importlib.util
key_maze_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'key-maze', 'key_door_maze_env.py')
spec = importlib.util.spec_from_file_location('key_door_maze_env', key_maze_path)
key_door_maze_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(key_door_maze_module)
KeyDoorMazeEnv = key_door_maze_module.KeyDoorMazeEnv

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simple DQN for small mazes
class SmallMazeNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(SmallMazeNetwork, self).__init__()
        
        # Instead of 8x8 kernel, use 3x3 kernel which is more appropriate for small mazes
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions
        def get_conv_output_size(shape):
            o = torch.zeros(1, *shape)
            o = F.relu(self.conv1(o))
            o = F.relu(self.conv2(o))
            return int(np.prod(o.size()))
        
        conv_out_size = get_conv_output_size(input_shape)
        
        self.fc = nn.Linear(conv_out_size, 64)
        self.out = nn.Linear(64, n_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.out(x)

# Simple DQN Agent
class SimpleDQNAgent:
    def __init__(self, state_shape, action_size, buffer_size=10000, 
                 learning_rate=0.001, gamma=0.99, batch_size=32):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Create Q networks
        self.q_network = SmallMazeNetwork(state_shape, action_size).to(device)
        self.target_network = SmallMazeNetwork(state_shape, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # For storing transitions
        self.Transition = namedtuple('Transition', 
                                    ('state', 'action', 'reward', 'next_state', 'done'))
        
    def remember(self, state, action, reward, next_state, done):
        # Convert to torch tensors and store
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
        self.memory.append(self.Transition(
            state, action, reward, next_state, done
        ))
    
    def act(self, state, eval_mode=False):
        # Convert state to torch tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Epsilon-greedy action selection
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
            
        # Sample a batch from memory
        transitions = random.sample(self.memory, self.batch_size)
        batch = self.Transition(*zip(*transitions))
        
        # Prepare batch
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.tensor(batch.action).to(device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)
        
        # Compute current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
        
        # Compute target Q values
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and update network
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def main():
    # Parameters
    maze_size = 10  # Increased to give more room
    num_keys = 1
    num_episodes = 5
    max_steps = 50
    
    # Create environment
    print("Creating Key-Door Maze environment...")
    env = KeyDoorMazeEnv(
        maze_size=maze_size,
        num_keys=num_keys,
        max_steps=max_steps,
        use_fixed_seed=True,
        fixed_seed=42
    )
    
    # Reset environment to get observation shape
    observation, info = env.reset()
    
    # Initialize agent
    # Important: Conv networks expect channels-first format
    observation_shape = (observation.shape[2], observation.shape[0], observation.shape[1])
    action_size = env.action_space.n
    
    print(f"Observation shape: {observation_shape}, Action size: {action_size}")
    
    agent = SimpleDQNAgent(
        state_shape=observation_shape,
        action_size=action_size
    )
    
    # Training loop
    for episode in range(num_episodes):
        observation, info = env.reset()
        
        # Transpose to channels-first for the neural network
        observation = np.transpose(observation, (2, 0, 1))
        
        total_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        while not done and step < max_steps:
            # Render environment
            env.render()
            
            # Select action
            action = agent.act(observation)
            
            # Take action
            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = np.transpose(next_observation, (2, 0, 1))  # Channels first
            
            # Store transition and train
            agent.remember(observation, action, reward, next_observation, done or truncated)
            agent.train()
            
            # Update observation
            observation = next_observation
            total_reward += reward
            step += 1
            
            print(f"Step {step}: Action {action}, Reward {reward:.2f}, Total {total_reward:.2f}")
            
            # Update target network periodically
            if step % 10 == 0:
                agent.update_target_network()
            
            if done or truncated:
                print(f"Episode {episode+1} finished after {step} steps with reward {total_reward:.2f}")
                
                # Check if goal was reached
                if info.get('goal_reached', False):
                    print("🎉 GOAL REACHED! 🎉")
                    
                # Print coverage stats
                coverage_stats = env.get_coverage_stats()
                print(f"Coverage: {coverage_stats['coverage_percentage']:.2f}%")
                break
    
    print("\nTest completed successfully!")
    
if __name__ == "__main__":
    main()
