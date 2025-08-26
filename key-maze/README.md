# Key-Door Maze Exploration Comparison

This folder contains the implementation of a Key-Door Maze environment and scripts to compare different exploration strategies:

1. Standard Q-learning (DQN)
2. Count-based exploration
3. Random Network Distillation (RND)
4. Rainbow DCEO

## Environment Description

The Key-Door Maze is a hierarchical exploration challenge where the agent must:
- Navigate through rooms divided by walls
- Find keys to unlock doors
- Reach the goal

This environment tests an agent's ability to perform complex, sequential exploration and is especially suited for evaluating hierarchical reinforcement learning approaches like Rainbow DCEO.

## Running the Comparison

To run the comparison with default settings:

```bash
python compare_keymazes.py
```

This will train and evaluate all four agents on the Key-Door Maze environment.

### Command Line Arguments

- `--episodes`: Number of episodes to train each agent (default: 50)
- `--eval_interval`: Evaluate every N episodes (default: 5)
- `--eval_episodes`: Number of evaluation episodes (default: 10)
- `--maze_size`: Size of the maze (default: 10)
- `--num_keys`: Number of keys in the maze (default: 2)
- `--max_steps`: Maximum steps per episode (default: 200)
- `--seed`: Random seed (default: 42)
- `--results_dir`: Results directory (default: "key_maze_results")
- `--skip`: Skip specific agents (e.g., `--skip standard count`)

## Example

For a more challenging environment with more keys:

```bash
python compare_keymazes.py --maze_size 12 --num_keys 3 --episodes 100
```

## Results

The script will generate comparison plots showing:
- Success rates
- Average rewards
- Average steps
- Keys collected
- Doors opened
- Environment coverage
- Training time

Results are saved to the specified results directory.
