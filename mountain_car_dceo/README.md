# Mountain Car DCEO Implementation

This directory contains an implementation of the fully online Deep Covering Option (DCEO) algorithm applied to the Mountain Car environment.

## Overview

The Mountain Car environment is an excellent test case for DCEO because:

1. **Exploration Challenge**: The car must first move away from the goal to build momentum, which requires discovering temporally extended behaviors (exactly what DCEO addresses).
2. **Sparse Reward Structure**: Mountain Car provides a -1 reward per step until the goal is reached (0 reward), making it challenging for standard RL.
3. **Continuous State Space**: Demonstrates DCEO's ability to handle continuous state spaces.
4. **Need for Temporally Extended Actions**: Success requires consistent back-and-forth movements to build momentum—perfect for the option-based approach.

## Files

- **train_mountain_car_dceo.py**: Main script to train the DCEO agent on Mountain Car
- **analyze_mountain_car_options.py**: Script to visualize and analyze the learned options
- **checkpoints/**: Directory where model checkpoints are saved during training

## Usage

### Training

To train the agent:

```
python train_mountain_car_dceo.py [--iterations 100] [--eval_freq 5] [--train_steps 10000] [--num_options 5] [--render]
```

Arguments:
- `--iterations`: Number of training iterations (default: 100)
- `--eval_freq`: Evaluation frequency in iterations (default: 5)
- `--train_steps`: Steps per training iteration (default: 10000)
- `--num_options`: Number of options to learn (default: 5)
- `--render`: Flag to render training (optional)

### Analyzing Options

To analyze the learned options after training:

```
python analyze_mountain_car_options.py --checkpoint path/to/checkpoint.pt
```

This will generate several visualizations:
- Option policies across the state space
- Option values (Q-values)
- Option usage patterns
- Learned representations
- Agent trajectories with option selection

## Expected Results

With DCEO on Mountain Car, you should expect:

1. **Efficient Exploration**: The agent should discover options that correspond to "rocking" behaviors (moving back and forth to build momentum)
2. **Interpretable Options**: The learned options might correspond to specific strategies like "move left", "move right", and possibly "build momentum"
3. **Temporal Abstraction**: The temporally extended nature of options should help the agent discover the solution more efficiently

## Requirements

The implementation uses the same dependencies as the main DCEO repository and requires:
- PyTorch
- Gym or Gymnasium (for Mountain Car environment)
- Matplotlib (for visualization)
- NumPy
- Seaborn (for plotting)
- scikit-learn (for t-SNE visualization)
