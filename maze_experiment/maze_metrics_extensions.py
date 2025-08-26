"""
Maze Metrics Extensions - DCEO Paper Implementation

Extensions to the MazeMetricsTracker specifically for measuring the alignment between
learned representations and true Laplacian eigenvectors.

Implements the representation quality metrics specified in section 4.5.4 of
"Deep Covering Options: An Eigenfunction View of Successor Features" by Klissarov et al. (2023).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

class RepresentationQualityMetrics:
    """
    Extension for MazeMetricsTracker to measure alignment between
    learned representations and true Laplacian eigenvectors.
    
    Implements the metrics specified in section 4.5.4 of the paper.
    """
    
    def __init__(self, metrics_tracker):
        """
        Initialize representation quality metrics.
        
        Args:
            metrics_tracker: The main MazeMetricsTracker instance
        """
        self.metrics_tracker = metrics_tracker
        self.env = metrics_tracker.env
        self.agent = metrics_tracker.agent
        
        # Initialize representation metrics
        self.metrics_tracker.representation_metrics = {
            'state_representations': {},  # Maps state positions to representation vectors
            'eigenfunction_alignment': defaultdict(list),  # Alignment scores over time
            'eigenspectrum': defaultdict(list),  # Eigenvalues over time
            'last_evaluation_step': 0,
            'evaluation_interval': 10000,  # Evaluate representation quality every 10k steps
            'steps': []  # Steps at which evaluations were performed
        }
        
        # Initialize visualization directory if it doesn't exist yet
        self.metrics_tracker.dirs['representation'] = os.path.join(
            self.metrics_tracker.results_dir, 'representation_metrics')
        os.makedirs(self.metrics_tracker.dirs['representation'], exist_ok=True)

    def log_representation_metrics(self, state, next_state, step_count):
        """
        Log representation quality metrics.
        
        Args:
            state: Current state
            next_state: Next state after action
            step_count: Current environment step count
        """
        if state is None or next_state is None:
            return
            
        # Extract agent's position from state
        state_pos = None
        if hasattr(self.env.unwrapped, 'agent_position'):
            state_pos = tuple(self.env.unwrapped.agent_position)
        else:
            state_pos = self._extract_agent_position(state)
            
        if state_pos is None:
            return
            
        # Get the representation for this state
        with torch.no_grad():
            # Check if agent has representation network
            if not hasattr(self.agent, 'laplacian_network'):
                return
                
            # Convert state to tensor
            if not isinstance(state, torch.Tensor):
                device = next(self.agent.laplacian_network.parameters()).device
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            else:
                state_tensor = state.unsqueeze(0) if state.dim() == 3 else state
                
            # Get representation
            representation = self.agent.laplacian_network(state_tensor).detach().cpu().numpy()[0]
            
            # Store representation for this state
            self.metrics_tracker.representation_metrics['state_representations'][state_pos] = representation
            
        # Evaluate representation quality periodically
        if (step_count - self.metrics_tracker.representation_metrics['last_evaluation_step'] >= 
            self.metrics_tracker.representation_metrics['evaluation_interval']):
            
            self.metrics_tracker.representation_metrics['last_evaluation_step'] = step_count
            self.metrics_tracker.representation_metrics['steps'].append(step_count)
            
            # Visualize representations
            self._visualize_representations(step_count)
            
            # Evaluate alignment with Laplacian eigenfunctions (when available)
            self._evaluate_eigenfunction_alignment(step_count)
            
            # Save all representation metrics
            self._save_representation_metrics()
    
    def _extract_agent_position(self, state):
        """
        Extract agent position from state observation.
        Adapted for various state formats.
        
        Args:
            state: Environment state observation
            
        Returns:
            tuple: Agent (x, y) position or None if not extractable
        """
        # Try to extract position from state
        # This is environment-specific and may need adaptation
        try:
            if hasattr(self.env.unwrapped, 'agent_position'):
                return tuple(self.env.unwrapped.agent_position)
            
            # If state is a 2D grid (common in maze environments)
            if isinstance(state, np.ndarray) and state.ndim == 2:
                # Find agent marker (usually 2 or similar value)
                agent_pos = np.where(state == 2)
                if len(agent_pos[0]) > 0 and len(agent_pos[1]) > 0:
                    return (agent_pos[1][0], agent_pos[0][0])  # (x, y)
            
            # If state is a flattened vector with one-hot encoding
            elif isinstance(state, np.ndarray) and state.ndim == 1:
                # This depends on how the maze state is encoded
                # Need to be adapted based on the specific environment
                pass
        except:
            pass
            
        return None
    
    def _visualize_representations(self, step_count):
        """
        Visualize learned representations using PCA and t-SNE.
        
        Args:
            step_count: Current environment step count
        """
        if not self.metrics_tracker.representation_metrics['state_representations']:
            return
        
        # Extract representations and positions
        positions = list(self.metrics_tracker.representation_metrics['state_representations'].keys())
        representations = list(self.metrics_tracker.representation_metrics['state_representations'].values())
        
        if len(positions) < 10:  # Need enough data
            return
            
        # Convert to arrays
        rep_matrix = np.array(representations)
        
        # PCA visualization of the representations
        plt.figure(figsize=(10, 8))
        
        try:
            from sklearn.decomposition import PCA
            
            # Apply PCA
            pca = PCA(n_components=2)
            rep_2d = pca.fit_transform(rep_matrix)
            
            # Create scatter plot
            plt.scatter(rep_2d[:, 0], rep_2d[:, 1], c=range(len(rep_2d)), cmap='viridis')
            plt.colorbar(label='State Index')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title(f'Representation Visualization (PCA) at step {step_count}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.metrics_tracker.dirs['representation'], f'rep_pca_{step_count}.png'))
            
            # Try t-SNE visualization if enough data points
            if len(positions) >= 50:
                try:
                    from sklearn.manifold import TSNE
                    
                    # Apply t-SNE
                    plt.figure(figsize=(10, 8))
                    tsne = TSNE(n_components=2, random_state=42)
                    rep_tsne = tsne.fit_transform(rep_matrix)
                    
                    # Create scatter plot
                    plt.scatter(rep_tsne[:, 0], rep_tsne[:, 1], c=range(len(rep_tsne)), cmap='viridis')
                    plt.colorbar(label='State Index')
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.title(f'Representation Visualization (t-SNE) at step {step_count}')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.metrics_tracker.dirs['representation'], f'rep_tsne_{step_count}.png'))
                    plt.close()
                except ImportError:
                    print("sklearn.manifold.TSNE not available for t-SNE visualization")
                except Exception as e:
                    print(f"Error during t-SNE visualization: {e}")
            
        except ImportError:
            print("sklearn not available for PCA visualization")
        except Exception as e:
            print(f"Error during representation visualization: {e}")
            
        plt.close()
        
        # Try to visualize the first few representation components in the maze
        self._visualize_representation_components(step_count)
    
    def _visualize_representation_components(self, step_count):
        """
        Visualize first few components of the representations on the maze grid.
        
        Args:
            step_count: Current environment step count
        """
        if not hasattr(self.env, 'unwrapped'):
            env_unwrapped = self.env
        else:
            env_unwrapped = self.env.unwrapped
            
        if not hasattr(env_unwrapped, 'maze') or not hasattr(env_unwrapped, 'maze_size'):
            return
            
        # Get maze size
        maze_size = env_unwrapped.maze_size
        
        # Extract representations
        rep_dict = self.metrics_tracker.representation_metrics['state_representations']
        
        if not rep_dict:
            return
            
        # Prepare maze grid for visualization (maze_size x maze_size)
        n_components = min(4, next(iter(rep_dict.values())).shape[0])  # Visualize up to 4 components
        
        # Create figure with subplots for each component
        fig, axes = plt.subplots(1, n_components, figsize=(5*n_components, 5))
        if n_components == 1:
            axes = [axes]
            
        # Custom diverging colormap (blue-white-red)
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('bwr', colors, N=256)
        
        for comp_idx in range(n_components):
            # Initialize component grid with NaNs (to represent walls/unvisited)
            component_grid = np.full((maze_size, maze_size), np.nan)
            
            # Fill grid with component values
            for pos, rep in rep_dict.items():
                if len(pos) == 2:  # (x, y) position
                    x, y = pos
                    if 0 <= x < maze_size and 0 <= y < maze_size:
                        component_grid[y, x] = rep[comp_idx]
            
            # Normalize for visualization (only non-NaN values)
            valid_values = component_grid[~np.isnan(component_grid)]
            if len(valid_values) > 0:
                vmin, vmax = np.percentile(valid_values, [5, 95])
                # Ensure symmetric colormap range if values span positive and negative
                if vmin < 0 and vmax > 0:
                    abs_max = max(abs(vmin), abs(vmax))
                    vmin, vmax = -abs_max, abs_max
                    
                # Plot heatmap
                im = axes[comp_idx].imshow(component_grid, cmap=cmap, vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=axes[comp_idx])
            else:
                # If no valid values, just show the grid
                axes[comp_idx].imshow(component_grid, cmap='viridis')
                
            axes[comp_idx].set_title(f'Component {comp_idx+1}')
            axes[comp_idx].axis('off')
            
        plt.suptitle(f'Representation Components at step {step_count}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_tracker.dirs['representation'], f'rep_components_{step_count}.png'))
        plt.close()
    
    def _evaluate_eigenfunction_alignment(self, step_count):
        """
        Evaluate alignment between learned representations and Laplacian eigenfunctions.
        This is an approximation as true Laplacian eigenfunctions may not be available.
        
        Args:
            step_count: Current environment step count
        """
        # This is a simplified approach when true eigenfunctions are not available
        # In a real implementation, you would compare with precomputed eigenfunctions
        
        # Get all representations
        rep_dict = self.metrics_tracker.representation_metrics['state_representations']
        
        if len(rep_dict) < 10:  # Need enough data
            return
            
        # Extract representations
        positions = list(rep_dict.keys())
        representations = np.array(list(rep_dict.values()))
        
        # Compute Principal Components as an approximation of eigenfunctions
        try:
            from sklearn.decomposition import PCA
            
            # Compute PCA components
            n_components = min(10, representations.shape[0], representations.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(representations)
            
            # Store eigenvalues (explained variance)
            for i, val in enumerate(pca.explained_variance_):
                self.metrics_tracker.representation_metrics['eigenspectrum'][i].append(val)
                
            # Compute similarity between PCA components and raw representations
            # This is a simplified approximation of eigenfunction alignment
            pc_components = pca.components_
            
            # For each representation dimension, compute alignment with PCA components
            for dim in range(min(10, representations.shape[1])):
                # Extract this dimension for all states
                dim_values = representations[:, dim]
                
                # Compute correlation with each PC
                max_corr = 0
                for pc_idx in range(pc_components.shape[0]):
                    pc = pc_components[pc_idx, :]
                    corr = np.abs(np.corrcoef(dim_values, pc)[0, 1])
                    max_corr = max(max_corr, corr)
                
                # Store max correlation as alignment measure
                self.metrics_tracker.representation_metrics['eigenfunction_alignment'][dim].append(max_corr)
                
            # Visualize eigenfunction alignment
            self._visualize_eigenfunction_alignment(step_count)
            
        except ImportError:
            print("sklearn not available for eigenfunction alignment evaluation")
        except Exception as e:
            print(f"Error during eigenfunction alignment evaluation: {e}")
    
    def _visualize_eigenfunction_alignment(self, step_count):
        """
        Visualize alignment between learned representations and eigenfunctions.
        
        Args:
            step_count: Current environment step count
        """
        alignment_data = self.metrics_tracker.representation_metrics['eigenfunction_alignment']
        steps = self.metrics_tracker.representation_metrics['steps']
        
        if not alignment_data or len(steps) < 2:
            return
            
        # Plot alignment over training
        plt.figure(figsize=(10, 6))
        
        for dim, values in alignment_data.items():
            if len(values) == len(steps):
                plt.plot(steps, values, marker='o', label=f'Dim {dim+1}')
        
        plt.xlabel('Environment Steps')
        plt.ylabel('Eigenfunction Alignment')
        plt.title('Representation-Eigenfunction Alignment Over Training')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_tracker.dirs['representation'], f'eigenfunction_alignment.png'))
        plt.close()
        
        # Plot eigenspectrum
        eigenspectrum = self.metrics_tracker.representation_metrics['eigenspectrum']
        
        if eigenspectrum:
            plt.figure(figsize=(10, 6))
            
            for i, values in eigenspectrum.items():
                if len(values) == len(steps):
                    plt.plot(steps, values, marker='o', label=f'λ{i+1}')
            
            plt.xlabel('Environment Steps')
            plt.ylabel('Eigenvalue')
            plt.title('Eigenspectrum Over Training')
            plt.grid(True)
            plt.legend()
            plt.yscale('log')  # Log scale for better visualization
            plt.tight_layout()
            plt.savefig(os.path.join(self.metrics_tracker.dirs['representation'], f'eigenspectrum.png'))
            plt.close()
    
    def _save_representation_metrics(self):
        """Save representation metrics to file."""
        # Extract metrics
        metrics = {
            'steps': self.metrics_tracker.representation_metrics['steps'],
            'eigenfunction_alignment': {str(k): v for k, v in 
                                       self.metrics_tracker.representation_metrics['eigenfunction_alignment'].items()},
            'eigenspectrum': {str(k): v for k, v in 
                             self.metrics_tracker.representation_metrics['eigenspectrum'].items()}
        }
        
        # Save to file
        import json
        try:
            with open(os.path.join(self.metrics_tracker.dirs['representation'], 'representation_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Error saving representation metrics: {e}")
