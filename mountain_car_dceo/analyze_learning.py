"""
Analysis script for DCEO Mountain Car learning progress.
This script extracts key metrics from checkpoint files and training logs to visualize
the agent's improvement over time, with a focus on exploring how the hyperparameter tuning
affected performance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import glob

def extract_metrics_from_logs(log_dir='./mountain_car_dceo/logs'):
    """Extract performance metrics from training logs."""
    print("Looking for log files...")
    
    # If log_dir doesn't exist, check if we have any training output files
    if not os.path.exists(log_dir):
        log_dir = '.'
    
    # Try to find any log files or training output with relevant data
    log_files = glob.glob(os.path.join(log_dir, '*.log')) + \
                glob.glob(os.path.join(log_dir, 'training_*.txt'))
    
    if not log_files:
        print("No log files found. Looking for any file with training information...")
        # Expanded search for any potential log files
        log_files = glob.glob(os.path.join(log_dir, '*log*')) + \
                    glob.glob(os.path.join(log_dir, '*output*')) + \
                    glob.glob(os.path.join(log_dir, '*train*'))
    
    print(f"Found {len(log_files)} potential log files")
    
    # Metrics to extract
    iterations = []
    rewards = []
    max_heights = []
    success_rates = []
    
    # Option usage statistics
    all_option_usage = {}
    
    # Parse log files
    for log_file in log_files:
        print(f"Examining {log_file}...")
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                
                # Look for evaluation results
                eval_patterns = [
                    r"Iteration (\d+)/\d+.*?Average Reward: ([+-]?\d+\.\d+).*?Average Max Height: ([+-]?\d+\.\d+).*?Success Rate: ([+-]?\d+\.\d+)",
                    r"Evaluation Results.*?Average Reward: ([+-]?\d+\.\d+).*?Success Rate: ([+-]?\d+\.\d+)",
                    r"Episode \d+: Reward = ([+-]?\d+\.\d+).*?Max Height = ([+-]?\d+\.\d+)"
                ]
                
                for pattern in eval_patterns:
                    matches = re.finditer(pattern, log_content, re.DOTALL)
                    for match in matches:
                        if len(match.groups()) == 4:  # Full pattern with iteration
                            iter_num = int(match.group(1))
                            reward = float(match.group(2))
                            height = float(match.group(3))
                            success = float(match.group(4))
                            
                            iterations.append(iter_num)
                            rewards.append(reward)
                            max_heights.append(height)
                            success_rates.append(success)
                        elif len(match.groups()) == 2:  # Just reward and success
                            reward = float(match.group(1))
                            success = float(match.group(2))
                            rewards.append(reward)
                            success_rates.append(success)
                
                # Look for option usage statistics
                option_usage_pattern = r"Option usage statistics:(.*?)(?=Step \d+|\Z)"
                option_matches = re.finditer(option_usage_pattern, log_content, re.DOTALL)
                
                for match in option_matches:
                    option_text = match.group(1)
                    
                    # Extract iteration from nearby text
                    step_matches = re.search(r"Step (\d+)/", log_content[:match.start()])
                    if step_matches:
                        step = int(step_matches.group(1))
                        # Approximate iteration based on steps (5000 steps per iteration)
                        approx_iter = (step // 5000) + 1
                        
                        # Extract option usage
                        usage_dict = {}
                        option_lines = re.finditer(r"Option (\d+): used (\d+) times, avg duration: ([+-]?\d+\.\d+)", option_text)
                        for opt_match in option_lines:
                            opt_id = int(opt_match.group(1))
                            count = int(opt_match.group(2))
                            duration = float(opt_match.group(3))
                            usage_dict[opt_id] = {'count': count, 'duration': duration}
                        
                        all_option_usage[approx_iter] = usage_dict
        
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    print(f"Extracted {len(iterations)} data points for iterations")
    print(f"Extracted option usage for {len(all_option_usage)} steps")
    
    return {
        'iterations': iterations,
        'rewards': rewards,
        'max_heights': max_heights,
        'success_rates': success_rates,
        'option_usage': all_option_usage
    }

def analyze_option_usage(option_usage):
    """Analyze how option usage patterns changed over time."""
    if not option_usage:
        print("No option usage data available")
        return None
    
    # Sort by iteration
    sorted_iterations = sorted(option_usage.keys())
    
    # Prepare data for plotting
    option_counts = {}
    option_durations = {}
    
    for iter_num in sorted_iterations:
        usage = option_usage[iter_num]
        
        for opt_id, stats in usage.items():
            if opt_id not in option_counts:
                option_counts[opt_id] = []
                option_durations[opt_id] = []
            
            option_counts[opt_id].append(stats['count'])
            option_durations[opt_id].append(stats['duration'])
    
    # Create plots for option usage
    if option_counts:
        num_options = len(option_counts)
        
        # Create plots directory
        plots_dir = './mountain_car_dceo/results/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot option counts
        plt.figure(figsize=(12, 6))
        for opt_id, counts in option_counts.items():
            # Pad with zeros if the list is shorter than the number of iterations
            padded_counts = counts + [0] * (len(sorted_iterations) - len(counts))
            plt.plot(sorted_iterations[:len(padded_counts)], padded_counts, marker='o', label=f'Option {opt_id}')
        
        plt.xlabel('Approx. Iteration')
        plt.ylabel('Usage Count')
        plt.title('Option Usage Frequency Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'option_usage_counts.png'))
        plt.close()
        
        # Plot option durations
        plt.figure(figsize=(12, 6))
        for opt_id, durations in option_durations.items():
            # Pad with zeros if the list is shorter than the number of iterations
            padded_durations = durations + [0] * (len(sorted_iterations) - len(durations))
            plt.plot(sorted_iterations[:len(padded_durations)], padded_durations, marker='o', label=f'Option {opt_id}')
        
        plt.xlabel('Approx. Iteration')
        plt.ylabel('Average Duration (steps)')
        plt.title('Option Duration Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'option_usage_durations.png'))
        plt.close()
        
        # Highlight transition point (iteration 8)
        transition_iter = 8
        
        # Analyze before and after transition
        before_indices = [i for i, iter_num in enumerate(sorted_iterations) if iter_num <= transition_iter]
        after_indices = [i for i, iter_num in enumerate(sorted_iterations) if iter_num > transition_iter]
        
        before_durations = {}
        after_durations = {}
        
        for opt_id, durations in option_durations.items():
            if len(durations) > 0:
                before_vals = [durations[i] for i in before_indices if i < len(durations)]
                after_vals = [durations[i] for i in after_indices if i < len(durations)]
                
                before_durations[opt_id] = np.mean(before_vals) if before_vals else 0
                after_durations[opt_id] = np.mean(after_vals) if after_vals else 0
        
        # Create a bar chart comparing option durations before and after tuning
        plt.figure(figsize=(10, 6))
        
        opt_ids = sorted(before_durations.keys())
        ind = np.arange(len(opt_ids))
        width = 0.35
        
        before_vals = [before_durations[opt_id] for opt_id in opt_ids]
        after_vals = [after_durations[opt_id] for opt_id in opt_ids]
        
        plt.bar(ind - width/2, before_vals, width, label='Before Tuning')
        plt.bar(ind + width/2, after_vals, width, label='After Tuning')
        
        plt.xlabel('Option ID')
        plt.ylabel('Average Duration (steps)')
        plt.title('Option Duration Before and After Hyperparameter Tuning')
        plt.xticks(ind, [f'Option {opt_id}' for opt_id in opt_ids])
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(plots_dir, 'option_duration_comparison.png'))
        plt.close()
        
        return {
            'before_tuning': before_durations,
            'after_tuning': after_durations
        }
    
    return None

def plot_learning_progress(metrics):
    """Plot the learning progress based on extracted metrics."""
    # Create plots directory
    plots_dir = './mountain_car_dceo/results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # If we have iteration-specific data
    if metrics['iterations']:
        # Sort data by iteration
        sorted_indices = np.argsort(metrics['iterations'])
        iterations = [metrics['iterations'][i] for i in sorted_indices]
        rewards = [metrics['rewards'][i] for i in sorted_indices]
        max_heights = [metrics['max_heights'][i] for i in sorted_indices]
        success_rates = [metrics['success_rates'][i] for i in sorted_indices]
        
        # Plot learning curves
        plt.figure(figsize=(15, 12))
        
        # Rewards
        plt.subplot(3, 1, 1)
        plt.plot(iterations, rewards, 'o-', linewidth=2)
        
        # Add vertical line at hyperparameter tuning point (iteration 8)
        plt.axvline(x=8, color='r', linestyle='--', label='Hyperparameter Tuning')
        
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Reward Progress')
        plt.legend()
        plt.grid(True)
        
        # Max Heights
        plt.subplot(3, 1, 2)
        plt.plot(iterations, max_heights, 'o-', linewidth=2)
        plt.axhline(y=0.5, color='g', linestyle='--', label='Goal Position')
        plt.axvline(x=8, color='r', linestyle='--', label='Hyperparameter Tuning')
        
        plt.xlabel('Iteration')
        plt.ylabel('Max Height')
        plt.title('Maximum Height Progress')
        plt.legend()
        plt.grid(True)
        
        # Success Rates
        plt.subplot(3, 1, 3)
        plt.plot(iterations, success_rates, 'o-', linewidth=2)
        plt.axvline(x=8, color='r', linestyle='--', label='Hyperparameter Tuning')
        
        plt.xlabel('Iteration')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'learning_progress.png'))
        plt.close()
        
        # Compare before and after tuning
        tuning_point = 8
        before_indices = [i for i, iter_num in enumerate(iterations) if iter_num <= tuning_point]
        after_indices = [i for i, iter_num in enumerate(iterations) if iter_num > tuning_point]
        
        if before_indices and after_indices:
            avg_reward_before = np.mean([rewards[i] for i in before_indices])
            avg_reward_after = np.mean([rewards[i] for i in after_indices])
            
            avg_height_before = np.mean([max_heights[i] for i in before_indices])
            avg_height_after = np.mean([max_heights[i] for i in after_indices])
            
            avg_success_before = np.mean([success_rates[i] for i in before_indices])
            avg_success_after = np.mean([success_rates[i] for i in after_indices])
            
            # Create comparison chart
            plt.figure(figsize=(12, 8))
            
            metrics_names = ['Average Reward', 'Max Height', 'Success Rate']
            before_values = [avg_reward_before, avg_height_before, avg_success_before]
            after_values = [avg_reward_after, avg_height_after, avg_success_after]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            plt.bar(x - width/2, before_values, width, label='Before Tuning')
            plt.bar(x + width/2, after_values, width, label='After Tuning')
            
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Performance Metrics Before and After Hyperparameter Tuning')
            plt.xticks(x, metrics_names)
            plt.legend()
            plt.grid(True, axis='y')
            
            # Add percentage improvement
            for i, (before, after) in enumerate(zip(before_values, after_values)):
                if before != 0:
                    percent_change = ((after - before) / abs(before)) * 100
                    plt.text(i, max(before, after) + 0.1, f"{percent_change:.1f}%", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'))
            plt.close()
            
            return {
                'avg_reward_before': avg_reward_before,
                'avg_reward_after': avg_reward_after,
                'avg_height_before': avg_height_before,
                'avg_height_after': avg_height_after,
                'avg_success_before': avg_success_before,
                'avg_success_after': avg_success_after
            }
    
    return None

def main():
    # Create results directory
    results_dir = './mountain_car_dceo/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract metrics from logs
    metrics = extract_metrics_from_logs()
    
    # Plot learning progress
    comparison = plot_learning_progress(metrics)
    
    # Analyze option usage
    option_analysis = analyze_option_usage(metrics['option_usage'])
    
    # Print summary if available
    if comparison:
        print("\nPerformance Comparison Before and After Hyperparameter Tuning:")
        print(f"Average Reward: {comparison['avg_reward_before']:.2f} → {comparison['avg_reward_after']:.2f}")
        print(f"Max Height: {comparison['avg_height_before']:.2f} → {comparison['avg_height_after']:.2f}")
        print(f"Success Rate: {comparison['avg_success_before']:.2f} → {comparison['avg_success_after']:.2f}")
    
    if option_analysis:
        print("\nOption Duration Changes:")
        for opt_id in sorted(option_analysis['before_tuning'].keys()):
            before = option_analysis['before_tuning'][opt_id]
            after = option_analysis['after_tuning'][opt_id]
            print(f"Option {opt_id}: {before:.2f} → {after:.2f} steps")
    
    print(f"\nAnalysis complete. Plots saved to {results_dir}/plots/")

if __name__ == '__main__':
    main()
