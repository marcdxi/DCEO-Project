"""
Debugging script for baselines with more detailed logging.
"""

import numpy as np
import torch
import os
import sys
from tqdm import tqdm
import traceback

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our environment and agent
from maze_experiment.complex_maze_env import ComplexMazeEnv
from my_baselines.rnd import RNDAgent
from my_baselines.utils import preprocess_state

def debug_observation_processing():
    """Debug observation processing from the maze environment."""
    print("\n=== DEBUGGING OBSERVATION PROCESSING ===")
    
    # Create environment and reset
    env = ComplexMazeEnv(maze_size=15)
    print(f"Created environment with observation space: {env.observation_space}")
    
    # Get initial observation
    initial_obs = env.reset()
    print(f"Initial observation type: {type(initial_obs)}")
    if hasattr(initial_obs, 'shape'):
        print(f"Initial observation shape: {initial_obs.shape}")
    
    # Take a step to get numpy array observation
    step_result = env.step(0)
    if len(step_result) == 5:
        next_obs, reward, done, truncated, info = step_result
    else:
        next_obs, reward, done, info = step_result
        
    print(f"Next observation type: {type(next_obs)}")
    if hasattr(next_obs, 'shape'):
        print(f"Next observation shape: {next_obs.shape}")
    
    # Try preprocessing the observation
    try:
        preprocessed = preprocess_state(next_obs)
        print(f"Preprocessed observation type: {type(preprocessed)}")
        print(f"Preprocessed observation shape: {preprocessed.shape}")
    except Exception as e:
        print(f"Error preprocessing observation: {e}")
        traceback.print_exc()
    
    return env, next_obs

def debug_agent_creation(env, observation):
    """Debug agent creation with the observation from the environment."""
    print("\n=== DEBUGGING AGENT CREATION ===")
    
    try:
        # Get input shape and number of actions
        if hasattr(observation, 'shape'):
            if len(observation.shape) == 3:  # Image observation (H, W, C)
                input_shape = (observation.shape[2], observation.shape[0], observation.shape[1])
                print(f"Using image input shape: {input_shape}")
            else:
                input_shape = observation.shape
                print(f"Using general input shape: {input_shape}")
        else:
            # Fallback to default
            input_shape = (3, 15, 15)
            print(f"Using default input shape: {input_shape}")
            
        num_actions = env.action_space.n
        print(f"Number of actions: {num_actions}")
        
        # Create the agent
        agent = RNDAgent(
            input_shape=input_shape,
            num_actions=num_actions,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=8,
            update_every=4,
            target_update=100,
            intrinsic_weight=0.5
        )
        print(f"Successfully created agent: {type(agent)}")
        
        return agent
    
    except Exception as e:
        print(f"Error creating agent: {e}")
        traceback.print_exc()
        return None

def debug_agent_step(env, agent, observation):
    """Debug a single agent step."""
    print("\n=== DEBUGGING AGENT STEP ===")
    
    try:
        # Select an action
        action = agent.select_action(observation)
        print(f"Selected action: {action}")
        
        # Take a step in the environment
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, done, truncated, info = step_result
            done = done or truncated
        else:
            next_obs, reward, done, info = step_result
            
        print(f"Received reward: {reward}, done: {done}")
        print(f"Info: {info}")
        
        # Process the step with the agent
        try:
            print("Processing step with agent...")
            agent.step(observation, action, reward, next_obs, done, episode=0)
            print("Step processed successfully!")
        except Exception as e:
            print(f"Error in agent.step: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error in debug_agent_step: {e}")
        traceback.print_exc()

def main():
    """Main debugging function."""
    print("Starting debugging process...")
    
    # Debug observation processing
    env, observation = debug_observation_processing()
    
    # Debug agent creation
    agent = debug_agent_creation(env, observation)
    
    if agent:
        # Debug agent step
        debug_agent_step(env, agent, observation)
    
    print("\nDebugging complete!")

if __name__ == "__main__":
    main()
