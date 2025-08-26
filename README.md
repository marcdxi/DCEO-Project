# Deep Covering Eigenoptions (DCEO) Framework

This repository contains an implementation of the Deep Covering Eigenoptions (DCEO) framework for hierarchical reinforcement learning, along with several baseline methods for comparison.

## Environments

The framework supports training agents in the following environments:

- **Key Maze**: A maze environment where the agent must first collect a key before reaching the goal.
- **Complex Maze**: A standard maze navigation environment with no keys, where the agent must find the goal.

## Algorithms

The following algorithms are implemented:

### DCEO (Deep Covering Eigenoptions)
- A hierarchical reinforcement learning approach that discovers useful options via eigendecomposition of the Laplacian.

### Baseline Methods
- **Tabular Q-Learning**: Classical Q-learning with tabular state representation.
- **DDQN with Count-Based Exploration**: Double Deep Q-Network with count-based exploration bonus.
- **RND (Random Network Distillation)**: Exploration through prediction error of a random neural network.
- **DQN**: Standard Deep Q-Network.

## Usage

### Training DCEO in Key Maze Environment

```bash
# Basic training with visualization
python train_keymaze_online_dceo_with_viz.py --maze_size 12 --iterations 2 --steps 500 --seed 44 --render --fixed_layout

# Training without visualization (faster)
python train_keymaze_online_dceo_with_viz.py --maze_size 12 --iterations 2 --steps 500 --seed 44 --fixed_layout

# Extended training (more iterations and steps)
python train_keymaze_online_dceo_with_viz.py --maze_size 12 --iterations 40 --steps 5000 --seed 44 --fixed_layout
