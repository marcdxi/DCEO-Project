"""
Test script to check environment observation format
"""

from maze_experiment.complex_maze_env import ComplexMazeEnv

# Create environment
env = ComplexMazeEnv(maze_size=15)

# Reset and get initial observation
obs = env.reset()

# Print observation type and shape
print(f"Observation type: {type(obs)}")
if hasattr(obs, 'shape'):
    print(f"Observation shape: {obs.shape}")

# Take a step
step_result = env.step(0)
print(f"Step result length: {len(step_result)}")
print(f"Next observation type: {type(step_result[0])}")
if hasattr(step_result[0], 'shape'):
    print(f"Next observation shape: {step_result[0].shape}")
