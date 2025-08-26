"""
Simple script to examine the performance metrics in checkpoint files.
This extracts basic information without requiring full agent loading.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import re

def extract_metrics_from_checkpoints(checkpoint_dir='./mountain_car_dceo/checkpoints'):
    """Extract basic performance metrics from checkpoint files."""
    # Create results directory if it doesn't exist
    os.makedirs('./mountain_car_dceo/results', exist_ok=True)
    
    # List checkpoint files
    checkpoint_files = sorted([
        os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
        if f.startswith('mountain_car_dceo_iter_') and f.endswith('.pt')
    ], key=lambda x: int(os.path.basename(x).split('_iter_')[1].split('.pt')[0]))
    
    # Add final checkpoint if it exists
    final_checkpoint = os.path.join(checkpoint_dir, 'mountain_car_dceo_final.pt')
    if os.path.exists(final_checkpoint):
        checkpoint_files.append(final_checkpoint)
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    all_results = []
    
    for checkpoint_path in checkpoint_files:
        print(f"\nExamining checkpoint: {checkpoint_path}")
        
        # Extract iteration number from filename
        match = re.search(r'iter_(\d+)', checkpoint_path)
        if match:
            iteration = int(match.group(1))
        else:
            # Final checkpoint
            iteration = max([int(re.search(r'iter_(\d+)', f).group(1)) for f in checkpoint_files if re.search(r'iter_(\d+)', f)]) + 1
        
        # Try to extract metadata from the checkpoint
        try:
            # Try to get file information at minimum
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Convert to MB
            print(f"  File size: {file_size:.2f} MB")
            
            # Create a simple record
            result = {
                'iteration': iteration,
                'file_path': checkpoint_path,
                'file_size_mb': file_size,
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error examining checkpoint: {e}")
    
    # Sort results by iteration
    all_results.sort(key=lambda x: x['iteration'])
    
    # Print summary
    print("\nSummary of All Checkpoints:")
    for result in all_results:
        print(f"Iteration {result['iteration']}: File size = {result['file_size_mb']:.2f} MB")
    
    return all_results

def analyze_checkpoint_timing(all_results):
    """Analyze the timing pattern of checkpoints to identify potential performance changes."""
    # Extract iterations and file sizes
    iterations = [r['iteration'] for r in all_results]
    file_sizes = [r['file_size_mb'] for r in all_results]
    
    # Plot file sizes by iteration (might indirectly indicate model complexity changes)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, file_sizes, 'o-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Checkpoint File Size (MB)')
    plt.title('Checkpoint File Size by Iteration')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./mountain_car_dceo/results/checkpoint_sizes.png')
    plt.close()
    
    print("Checkpoint size analysis saved to ./mountain_car_dceo/results/checkpoint_sizes.png")
    
    # Analyze iteration patterns
    print("\nIteration Pattern Analysis:")
    
    # Look for jumps in iteration numbers that might indicate the start of a new hyperparameter set
    iter_diffs = [iterations[i+1] - iterations[i] for i in range(len(iterations)-1)]
    mean_diff = np.mean(iter_diffs)
    std_diff = np.std(iter_diffs)
    
    print(f"Average iteration gap: {mean_diff:.2f}")
    
    # Find any pattern disruptions that might indicate hyperparameter changes
    for i, diff in enumerate(iter_diffs):
        if diff > mean_diff + 2*std_diff:
            print(f"Potential checkpoint pattern change detected between iterations {iterations[i]} and {iterations[i+1]}")
            print(f"  Gap of {diff} iterations (more than 2 standard deviations from mean)")
    
    # If we have many checkpoints, try to identify before/after tuning patterns
    if len(iterations) >= 6:
        early_iter = iterations[:len(iterations)//2]
        later_iter = iterations[len(iterations)//2:]
        
        early_gaps = [early_iter[i+1] - early_iter[i] for i in range(len(early_iter)-1)]
        later_gaps = [later_iter[i+1] - later_iter[i] for i in range(len(later_iter)-1)]
        
        if early_gaps and later_gaps:
            print(f"\nEarly iterations ({early_iter[0]}-{early_iter[-1]}) average gap: {np.mean(early_gaps):.2f}")
            print(f"Later iterations ({later_iter[0]}-{later_iter[-1]}) average gap: {np.mean(later_gaps):.2f}")
    
    # Based on how the checkpoints were created after hyperparameter tuning
    if len(iterations) > 8:
        transition_point = 8  # Assume hyperparameter tuning after iteration 8
        before_tuning = [r for r in all_results if r['iteration'] <= transition_point]
        after_tuning = [r for r in all_results if r['iteration'] > transition_point]
        
        print("\nCheckpoint Analysis Based on Assumed Hyperparameter Tuning Point:")
        print(f"Before tuning (iterations 1-{transition_point}): {len(before_tuning)} checkpoints")
        print(f"After tuning (iterations >{transition_point}): {len(after_tuning)} checkpoints")
        
        if before_tuning and after_tuning:
            avg_size_before = np.mean([r['file_size_mb'] for r in before_tuning])
            avg_size_after = np.mean([r['file_size_mb'] for r in after_tuning])
            
            print(f"Average checkpoint size before tuning: {avg_size_before:.2f} MB")
            print(f"Average checkpoint size after tuning: {avg_size_after:.2f} MB")
            
            if avg_size_after > avg_size_before:
                print("Checkpoint size increased after tuning, suggesting potential model capacity increase")
            elif avg_size_after < avg_size_before:
                print("Checkpoint size decreased after tuning")
            else:
                print("Checkpoint size remained similar after tuning")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract basic metrics from checkpoint files')
    parser.add_argument('--checkpoint_dir', type=str, default='./mountain_car_dceo/checkpoints',
                        help='Directory containing checkpoint files')
    
    args = parser.parse_args()
    
    all_results = extract_metrics_from_checkpoints(checkpoint_dir=args.checkpoint_dir)
    analyze_checkpoint_timing(all_results)

if __name__ == '__main__':
    main()
