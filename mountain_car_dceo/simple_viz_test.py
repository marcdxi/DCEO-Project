"""
Simplified training script that focuses on clear visualization of Mountain Car + DCEO.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt
import gymnasium as gym
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('mountain_car_viz')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def main():
    """Simple visualization test with the Mountain Car environment."""
    
    # Create environment
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    logger.info(f"Created Mountain Car environment")
    
    # Create figure
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title("Mountain Car Visualization")
    
    # Make figure visible on top
    fig.canvas.manager.window.title("Mountain Car Visualization - DCEO Test")
    
    # Try to make window visible
    try:
        # Make window appear on top
        fig.canvas.manager.window.attributes('-topmost', 1)
        # Disable topmost after bringing to front
        fig.canvas.manager.window.attributes('-topmost', 0)
    except Exception as e:
        logger.warning(f"Could not bring window to front: {e}")
    
    plt.show(block=False)
    
    # Run simulation
    state, _ = env.reset()
    done = False
    step_count = 0
    
    logger.info("Starting simulation loop")
    
    # Show window for 60 seconds maximum
    start_time = time.time()
    while time.time() - start_time < 60:
        # Take a random action
        action = env.action_space.sample()
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        # Render environment
        img = env.render()
        
        # Update plot
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Mountain Car - Step: {step_count}, Position: {next_state[0]:.3f}, Velocity: {next_state[1]:.3f}")
        
        # Log for debugging
        if step_count % 10 == 0:
            logger.info(f"Step {step_count}: Position = {next_state[0]:.3f}, Velocity = {next_state[1]:.3f}")
        
        # Update display
        try:
            plt.draw()
            plt.pause(0.05)  # Longer pause to make visualization more visible
        except Exception as e:
            logger.error(f"Error updating display: {e}")
        
        # Move to next state
        state = next_state
        step_count += 1
        
        # Reset if done
        if done:
            state, _ = env.reset()
            logger.info(f"Episode finished after {step_count} steps. Resetting.")
    
    logger.info("Visualization test complete")
    plt.ioff()
    plt.close()
    env.close()

if __name__ == "__main__":
    logger.info("Starting visualization test...")
    main()
