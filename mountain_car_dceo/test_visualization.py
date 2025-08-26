"""
Simple test script to verify visualization works correctly in the Mountain Car environment.
This script only tests the rendering functionality without the full DCEO algorithm.
"""

import os
import sys
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('visualization_test')

# Try to import gym
try:
    import gymnasium as gym
    logger.info("Using Gymnasium API")
    GYM_TYPE = "gymnasium" 
except ImportError:
    try:
        import gym
        logger.info("Using OpenAI Gym API")
        GYM_TYPE = "gym"
    except ImportError:
        logger.error("Neither gymnasium nor gym is installed!")
        sys.exit(1)

def test_matplotlib():
    """Test if matplotlib can create and display a simple plot."""
    logger.info("Testing matplotlib basic functionality...")
    
    try:
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro-')
        plt.title('Simple Test Plot')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
        logger.info("Created test matplotlib figure")
        
        # Try to draw the figure
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()
        logger.info("Successfully rendered test plot")
        
        time.sleep(2)  # Keep the plot visible for 2 seconds
        return True
    except Exception as e:
        logger.error(f"Error creating test plot: {e}")
        return False

def preprocess_state(state):
    """Preprocess the state from the Mountain Car environment for visualization."""
    
    # Handle gym's new API which returns (obs, info) tuples
    if isinstance(state, tuple):
        state = state[0]  # Extract observation
    
    # Convert raw MountainCar state to a visual representation
    # state[0] is position (-1.2 to 0.6)
    # state[1] is velocity (-0.07 to 0.07)
    
    # Create a simple 84x84 visual representation
    visual_state = np.zeros((3, 84, 84), dtype=np.float32)
    
    # Normalize position to 0-1 range
    pos_normalized = (state[0] + 1.2) / 1.8  # Map [-1.2, 0.6] to [0, 1]
    
    # Normalize velocity to 0-1 range
    vel_normalized = (state[1] + 0.07) / 0.14  # Map [-0.07, 0.07] to [0, 1]
    
    # Create visual representation:
    # Channel 0: Position (horizontal bar)
    pos_pixel = int(pos_normalized * 83)
    visual_state[0, 41:43, :] = 0.3  # Draw the "track"
    visual_state[0, 35:48, pos_pixel-2:pos_pixel+3] = 1.0  # Car position
    
    # Channel 1: Velocity (color intensity)
    visual_state[1] = vel_normalized * np.ones((84, 84))
    
    # Channel 2: Goal location marker
    goal_pixel = int((0.6 + 1.2) / 1.8 * 83)
    visual_state[2, 30:55, goal_pixel-4:goal_pixel+4] = 1.0
    
    return visual_state

def handle_env_reset(env, **kwargs):
    """Universal reset function that works with both old and new Gym APIs."""
    reset_result = env.reset(**kwargs)
    
    # Handle different reset() return formats
    if isinstance(reset_result, tuple):
        if len(reset_result) == 2:  # New Gymnasium API: (state, info)
            state, _ = reset_result
            return state
        else:
            return reset_result[0]  # Just return the first element
    else:  # Old Gym API: state
        return reset_result

def handle_env_step(env, action):
    """Universal step function that works with both old and new Gym APIs."""
    step_result = env.step(action)
    
    # Handle different step() return formats
    if isinstance(step_result, tuple):
        if len(step_result) == 5:  # New Gymnasium API: (next_state, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            return next_state, reward, done, info
        else:  # Old Gym API: (next_state, reward, done, info)
            return step_result
    else:
        raise ValueError(f"Unexpected step result type: {type(step_result)}")

def test_mountain_car_rendering():
    """Test if Mountain Car environment can be properly rendered."""
    logger.info("Testing Mountain Car environment rendering...")
    
    try:
        # Create environment
        if GYM_TYPE == "gymnasium":
            env = gym.make('MountainCar-v0', render_mode='rgb_array')
        else:
            env = gym.make('MountainCar-v0')
        
        logger.info(f"Created Mountain Car environment with state space: {env.observation_space}")
        
        # Reset the environment
        state = handle_env_reset(env)
        logger.info(f"Reset environment, initial state: {state}")
        
        # Create figure for rendering
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.ion()  # Interactive mode
        plt.show(block=False)
        
        # Test rendering capability
        success = False
        for i in range(100):  # Take 100 random actions
            # Take random action
            action = env.action_space.sample()
            next_state, reward, done, info = handle_env_step(env, action)
            
            # Attempt to render
            try:
                if GYM_TYPE == "gymnasium":
                    mc_img = env.render()
                else:
                    mc_img = env.render(mode='rgb_array')
                
                if mc_img is not None:
                    logger.info(f"Successfully rendered frame {i}, shape: {mc_img.shape}")
                    
                    # Display image
                    ax.clear()
                    ax.imshow(mc_img)
                    ax.set_title(f"Mountain Car - Step {i}, State: {next_state}")
                    
                    try:
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        logger.info("Successfully updated the display")
                        success = True
                    except Exception as e:
                        logger.error(f"Error updating display: {e}")
                
                # Also test the preprocessed state visualization
                preprocessed = preprocess_state(next_state)
                logger.info(f"Preprocessed state shape: {preprocessed.shape}")
                
            except Exception as e:
                logger.error(f"Error rendering Mountain Car: {e}")
            
            # Short delay
            time.sleep(0.05)
            
            # Move to next state
            state = next_state
            
            if done:
                logger.info("Episode finished, resetting environment")
                state = handle_env_reset(env)
        
        env.close()
        return success
    
    except Exception as e:
        logger.error(f"Error in Mountain Car test: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting visualization tests...")
    
    # Test matplotlib
    matplotlib_works = test_matplotlib()
    logger.info(f"Matplotlib test {'passed' if matplotlib_works else 'failed'}")
    
    # Test mountain car
    mountain_car_works = test_mountain_car_rendering()
    logger.info(f"Mountain Car rendering test {'passed' if mountain_car_works else 'failed'}")
    
    # Overall assessment
    if matplotlib_works and mountain_car_works:
        logger.info("All visualization tests passed!")
    else:
        logger.warning("Some visualization tests failed. Check the logs for details.")

if __name__ == "__main__":
    main()
