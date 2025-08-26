"""
Key Maze Metrics Extensions - Representation Quality Metrics

This module provides an extension to the KeyMazeMetricsTracker for measuring 
alignment between learned representations and true Laplacian eigenvectors.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import scipy.sparse as sp
import scipy.sparse.linalg
import os
from sklearn.metrics.pairwise import cosine_similarity

class RepresentationQualityMetrics:
    """Extension class for representation quality metrics in the Key Maze environment."""
    
    def __init__(self, metrics_tracker):
        """Initialize representation quality metrics.
        
        Args:
            metrics_tracker: The KeyMazeMetricsTracker instance to extend
        """
        self.metrics_tracker = metrics_tracker
        self.env = metrics_tracker.env
        self.agent = metrics_tracker.agent
        self.rep_dim = getattr(self.agent, 'rep_dim', 64)  # Get rep_dim from agent or use default
        
        # Initialize representation metrics
        self.init_representation_metrics()
        
    def init_representation_metrics(self):
        """Initialize representation quality metrics."""
        self.metrics_tracker.representation_metrics = {
            # Alignment with true eigenvectors
            'true_eigenvectors': None,  # Will be computed once we have enough state samples
            'true_eigenvalues': None,
            'feature_eigenvector_alignment': [],  # Cosine similarity between learned features and true eigenvectors
            
            # Representation data collection
            'state_representations': {},  # Maps state positions to representation vectors
            'visited_states_matrix': None,  # Matrix of all visited states' representations
            
            # Evaluation timing
            'last_evaluation_step': 0,
            'evaluation_interval': 10000  # Evaluate representation quality every 10k steps
        }
        
        # Create representation directory
        if 'representation' not in self.metrics_tracker.dirs:
            self.metrics_tracker.dirs['representation'] = os.path.join(
                self.metrics_tracker.results_dir, "representation")
            os.makedirs(self.metrics_tracker.dirs['representation'], exist_ok=True)
    
    def log_representation_metrics(self, state, next_state, step_count):
        """Log representation quality metrics for a step.
        
        This method records state representations and periodically evaluates
        the alignment with true eigenvectors.
        
        Args:
            state: Current state observation
            next_state: Next state observation
            step_count: Global step counter
        """
        # Skip if we don't have valid states
        if state is None or next_state is None:
            return
            
        # Extract agent's position from state
        state_pos = self._extract_agent_position(state)
        if state_pos is None:
            return
            
        # Get the representation for this state using the agent's representation network
        with torch.no_grad():
            # Convert state to tensor if it's not already
            if not isinstance(state, torch.Tensor):
                # Preprocess state if needed (convert to tensor and add batch dimension)
                if hasattr(self.agent, 'laplacian_network'):
                    # FullyOnlineDCEOAgent
                    device = next(self.agent.laplacian_network.parameters()).device
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                elif hasattr(self.agent, 'rep_net'):
                    # KeyMazeDCEOAgent
                    device = next(self.agent.rep_net.parameters()).device
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                else:
                    print("Warning: Could not find representation network in agent")
                    return
            else:
                state_tensor = state.unsqueeze(0) if state.dim() == 3 else state
                
            # Get representation from agent's network
            try:
                if hasattr(self.agent, 'laplacian_network'):
                    # FullyOnlineDCEOAgent
                    representation = self.agent.laplacian_network(state_tensor).detach().cpu().numpy()[0]
                elif hasattr(self.agent, 'rep_net'):
                    # KeyMazeDCEOAgent
                    rep_output = self.agent.rep_net(state_tensor)
                    representation = rep_output['rep'][0].cpu().numpy()
                else:
                    print("Warning: Could not find representation network in agent")
                    return
            except Exception as e:
                print(f"Error getting representation: {e}")
                return
            
            # Store the representation for this state (convert tuple to string for JSON compatibility)
            state_key = f"{state_pos[0]},{state_pos[1]}"
            self.metrics_tracker.representation_metrics['state_representations'][state_key] = representation
        
        # Periodically evaluate representation quality
        evaluation_interval = self.metrics_tracker.representation_metrics['evaluation_interval']
        last_eval_step = self.metrics_tracker.representation_metrics['last_evaluation_step']
        
        if (step_count % evaluation_interval == 0 or step_count == 1) and \
           step_count > last_eval_step + evaluation_interval // 2:
            self.evaluate_representation_quality(step_count)
            self.metrics_tracker.representation_metrics['last_evaluation_step'] = step_count
    
    def evaluate_representation_quality(self, step_count):
        """Evaluate representation quality based on alignment with true eigenvectors.
        
        Args:
            step_count: Global step counter
        """
        print(f"\n[DEBUG] Evaluating representation quality at step {step_count}")
        # Skip if we don't have enough state representations
        n_states = len(self.metrics_tracker.representation_metrics['state_representations'])
        print(f"[DEBUG] Found {n_states} state representations")
        if n_states < 10:
            print("[DEBUG] Not enough states to evaluate representation quality")
            return
        print(f"[DEBUG] Proceeding with evaluation")
            
        # Compute true Laplacian eigenvectors if needed
        if self.metrics_tracker.representation_metrics['true_eigenvectors'] is None:
            self._compute_true_laplacian_eigenvectors()
            
        # Evaluate alignment with true eigenvectors
        if self.metrics_tracker.representation_metrics['true_eigenvectors'] is not None:
            self._evaluate_eigenvector_alignment()
    
    def _compute_true_laplacian_eigenvectors(self):
        """Compute the true Laplacian eigenvectors of the state transition graph."""
        print("\n[DEBUG] Attempting to compute true Laplacian eigenvectors...")
        
        # Get only visited states from our collected representations
        print("[DEBUG] Extracting visited states from collected representations")
        state_reps = self.metrics_tracker.representation_metrics['state_representations']
        visited_states = []
        visited_reps = []
        
        for state_key, rep in state_reps.items():
            # Convert string key back to tuple
            state = tuple(map(int, state_key.split(',')))
            visited_states.append(state)
            visited_reps.append(rep)
        
        n_states = len(visited_states)
        print(f"[DEBUG] Using {n_states} actually visited states")
        
        # Check if we have enough states for a meaningful analysis
        if n_states < 10:
            print(f"[DEBUG] Not enough visited states for meaningful analysis: {n_states} < 10")
            return
        
        # Build representation matrix from visited states only
        rep_matrix = np.array(visited_reps)
        rep_dim = rep_matrix.shape[1] if rep_matrix.shape[0] > 0 else 0
        
        print(f"[DEBUG] Built representation matrix of shape {rep_matrix.shape}")
        
        # Store the visited states and their representations
        self.metrics_tracker.representation_metrics['rep_matrix'] = rep_matrix
        self.metrics_tracker.representation_metrics['states'] = visited_states
        
        # Now create an adjacency matrix for these visited states
        print(f"[DEBUG] Computing eigenvectors for {n_states} visited states")
        
        # Create adjacency matrix based on grid structure (much simpler/faster)
        adjacency = np.zeros((n_states, n_states))
        print("[DEBUG] Building adjacency matrix for visited states...")
        
        # Two states are adjacent if they're neighboring cells in the grid
        # Use Manhattan distance = 1 to determine adjacent states
        for i, state1 in enumerate(visited_states):
            for j, state2 in enumerate(visited_states):
                if i != j:
                    # Manhattan distance of 1 means adjacent
                    if abs(state1[0] - state2[0]) + abs(state1[1] - state2[1]) == 1:
                        adjacency[i, j] = 1.0
        
        # Create sparse matrix for efficiency
        adjacency_sparse = sp.csr_matrix(adjacency)
        connections = adjacency_sparse.count_nonzero()
        print(f"[DEBUG] Built adjacency matrix with {connections} connections between visited states")
        
        # Compute Laplacian matrix  
        degrees = np.array(adjacency_sparse.sum(axis=1)).flatten()
        degree_matrix = sp.diags(degrees)
        laplacian = degree_matrix - adjacency_sparse
        # Compute eigenvectors of the Laplacian
        try:
            # Compute the k smallest eigenvalues and corresponding eigenvectors
            print(f"[DEBUG] Computing eigenvectors...")
            
            # Ensure we compute at least 5 eigenvectors or as many as the rep dimension
            k_eigenvectors = min(max(rep_dim, 5), n_states-1)
            
            # Check for a degenerate case where we don't have enough connected states
            if np.sum(degrees) == 0 or n_states <= 1:
                print("[DEBUG] Cannot compute eigenvectors - not enough connected states")
                return
                
            # Try multiple approaches to compute eigenvectors
            # Starting with the most robust methods
            # Using try/except to handle potential numerical issues
            eigensolver_success = False
            
            try:
                # Try most robust approach first
                negated_laplacian = -laplacian
                eigenvalues, eigenvectors = sp.linalg.eigs(negated_laplacian, k=k_eigenvectors, which='LM', maxiter=2000)
                eigenvalues = -eigenvalues  # Negate back
                eigensolver_success = True
                print("[DEBUG] Used negated Laplacian with LM for eigendecomposition")
            except Exception as e1:
                print(f"[DEBUG] Error with negated Laplacian approach: {e1}")
                
                try:
                    # Try a different eigensolver - scipy's older method
                    eigenvalues, eigenvectors = sp.linalg.eigsh(laplacian, k=k_eigenvectors, which='SM', maxiter=2000)
                    eigensolver_success = True
                    print("[DEBUG] Used eigsh for eigendecomposition")
                except Exception as e2:
                    print(f"[DEBUG] Error with eigsh approach: {e2}")
                    
                    # Last resort - use dense SVD
                    try:
                        print("[DEBUG] Trying dense matrix approach")
                        lap_dense = laplacian.toarray()
                        # Use numpy's eigh for symmetric matrices (Laplacian is symmetric)
                        eigenvalues, eigenvectors = np.linalg.eigh(lap_dense)
                        # Only keep k smallest eigenvalues
                        idx = np.argsort(eigenvalues)[:k_eigenvectors]
                        eigenvalues = eigenvalues[idx]
                        eigenvectors = eigenvectors[:, idx]
                        eigensolver_success = True
                        print("[DEBUG] Used dense matrix eigendecomposition")
                    except Exception as e3:
                        print(f"[DEBUG] All eigensolvers failed - final error: {e3}")
            
            if not eigensolver_success:
                print("[DEBUG] Could not compute eigenvectors - cannot evaluate representation quality")
                return
            
            print(f"[DEBUG] Successfully computed {len(eigenvalues)} eigenvectors")
            
            # Sort by eigenvalue (smallest to largest)
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx].real  # Take real part
            eigenvectors = eigenvectors[:, idx].real  # Take real part
            
            # Normalize eigenvectors for more stable comparison
            for i in range(eigenvectors.shape[1]):
                norm = np.linalg.norm(eigenvectors[:, i])
                if norm > 1e-10:
                    eigenvectors[:, i] = eigenvectors[:, i] / norm
            
            # Print eigenvalue statistics to verify they're reasonable
            print(f"[DEBUG] Eigenvalue stats: min={np.min(eigenvalues)}, max={np.max(eigenvalues)}, mean={np.mean(eigenvalues)}")
            
            # Store for later use
            self.metrics_tracker.representation_metrics['true_eigenvalues'] = eigenvalues
            self.metrics_tracker.representation_metrics['true_eigenvectors'] = eigenvectors
            self.metrics_tracker.representation_metrics['rep_matrix'] = rep_matrix  # Store for plotting
            self.metrics_tracker.representation_metrics['states'] = visited_states  # Store state coordinates
            print(f"[DEBUG] Stored eigenvalues: {eigenvalues[:5]}")
            print(f"[DEBUG] Stored eigenvectors shape: {eigenvectors.shape}")
        except Exception as e:
            print(f"[DEBUG] Error computing true Laplacian eigenvectors: {e}")
            import traceback
            traceback.print_exc()
    
    def _are_adjacent_states(self, state1, state2):
        """Check if two states are adjacent (differ by one step)."""
        # For grid positions, manhattan distance of 1 means adjacent
        try:
            return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1]) == 1
        except Exception as e:
            print(f"[CRITICAL] Error in _are_adjacent_states: {e}")
            print(f"state1: {state1}, state2: {state2}")
            return False
    
    def _evaluate_eigenvector_alignment(self):
        """Evaluate alignment between learned features and true Laplacian eigenvectors."""
        print("\n[DEBUG] Starting eigenvector alignment evaluation")
        
        # Check if we have the necessary data
        req_keys = ['rep_matrix', 'true_eigenvectors', 'true_eigenvalues']
        for key in req_keys:
            if key not in self.metrics_tracker.representation_metrics:
                print(f"[DEBUG] Missing required data: {key}")
                return
            elif self.metrics_tracker.representation_metrics[key] is None:
                print(f"[DEBUG] Required data is None: {key}")
                return
            
        # Get feature representations and true eigenvectors
        rep_matrix = self.metrics_tracker.representation_metrics['rep_matrix']
        true_eigenvectors = self.metrics_tracker.representation_metrics['true_eigenvectors']
        
        # Center the data for more accurate alignment measurement
        # This removes any common bias across states and improves correlation measurement
        rep_matrix_centered = rep_matrix - np.mean(rep_matrix, axis=0, keepdims=True)
        
        print(f"[DEBUG] Computing alignment for {rep_matrix.shape[0]} states with {rep_matrix.shape[1]} features")
        print(f"[DEBUG] True eigenvectors shape: {true_eigenvectors.shape}")
        
        # Compute correlation between each feature and each eigenvector
        n_features = min(rep_matrix.shape[1], true_eigenvectors.shape[1])
        alignment_matrix = np.zeros((n_features, n_features))
        
        # First, ensure we're comparing a reasonable number of features
        n_features = min(20, n_features)  # Limit to 20 features maximum
        
        for i in range(n_features):
            # Extract feature i across all states
            feature_i = rep_matrix_centered[:, i]
            
            for j in range(n_features):
                eigenvector_j = true_eigenvectors[:, j]
                
                # Only measure alignment if both vectors have non-zero norm
                norm_f = np.linalg.norm(feature_i)
                norm_e = np.linalg.norm(eigenvector_j)
                
                if norm_f > 1e-6 and norm_e > 1e-6:
                    # Try both signs of the eigenvector and take the max correlation
                    # (eigenvectors can be flipped while preserving their meaning)
                    corr_pos = np.abs(np.dot(feature_i, eigenvector_j) / (norm_f * norm_e))
                    corr_neg = np.abs(np.dot(feature_i, -eigenvector_j) / (norm_f * norm_e))
                    alignment = max(corr_pos, corr_neg)
                else:
                    alignment = 0.0
                    
                alignment_matrix[i, j] = alignment
        
        # Compute a secondary alignment excluding the trivial eigenvector
        # (Eigenvector 0 is usually the constant eigenvector and less informative)
        alignment_matrix_nonconstant = alignment_matrix.copy()
        if alignment_matrix.shape[1] > 1:
            alignment_matrix_nonconstant[:, 0] = 0
        
        # Find best matching eigenvector for each feature
        best_matches = []
        for i in range(n_features):
            # Look for best match among non-constant eigenvectors if possible
            if np.max(alignment_matrix_nonconstant[i, :]) > 0.3:  # Reasonable threshold
                best_idx = np.argmax(alignment_matrix_nonconstant[i, :])
            else:  # Fall back to any eigenvector
                best_idx = np.argmax(alignment_matrix[i, :])
                
            best_value = alignment_matrix[i, best_idx]
            best_matches.append((i, best_idx, best_value))
        
        # Sort by alignment value
        best_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Store results
        self.metrics_tracker.representation_metrics['alignment_matrix'] = alignment_matrix
        self.metrics_tracker.representation_metrics['best_matches'] = best_matches
        
        print(f"[DEBUG] Successfully computed eigenvector alignment")
        # Log top alignments
        print("[DEBUG] Top alignments (feature, eigenvector, alignment):")
        for i, (feat, eigen, align) in enumerate(best_matches[:min(5, len(best_matches))]):
            print(f"[DEBUG]   {i+1}. Feature {feat} <-> Eigenvector {eigen}: {align:.4f}")
    
    def _extract_agent_position(self, state):
        """Extract agent position from state representation.
        
        Args:
            state: State observation
            
        Returns:
            Tuple (x, y) of agent position, or None if not extractable
        """
        # First, check if environment has agent_position attribute
        if hasattr(self.env.unwrapped, 'agent_position'):
            return tuple(self.env.unwrapped.agent_position)
            
        # If not, try to extract from observation
        try:
            if isinstance(state, tuple) and len(state) > 0:
                state = state[0]  # Handle (state, info) format
                
            if isinstance(state, np.ndarray):
                # For multi-channel grid representation (channel 0 is agent)
                if len(state.shape) == 3 and state.shape[0] >= 1:
                    agent_channel = state[0]  # Assuming first channel is agent
                    agent_pos = np.where(agent_channel > 0)
                    if len(agent_pos[0]) > 0 and len(agent_pos[1]) > 0:
                        return (agent_pos[0][0], agent_pos[1][0])
                # Older format where channels are the last dimension
                elif len(state.shape) == 3 and state.shape[2] >= 1:
                    agent_channel = state[:, :, 0]
                    agent_pos = np.where(agent_channel > 0)
                    if len(agent_pos[0]) > 0 and len(agent_pos[1]) > 0:
                        return (agent_pos[0][0], agent_pos[1][0])
        except Exception as e:
            print(f"Error extracting agent position: {e}")
            
        return None
    
    def plot_representation_quality(self, final=False):
        """Plot representation quality metrics focusing on eigenvector alignment.
        
        Args:
            final: Whether this is the final plot after training (True) or a milestone plot (False)
        """
        print("\n[DEBUG] Attempting to plot representation quality metrics")
        
        # Check if we have the necessary data for plotting
        required_keys = ['alignment_matrix', 'best_matches', 'rep_matrix', 'states']
        missing_keys = [key for key in required_keys if key not in self.metrics_tracker.representation_metrics]
        
        if missing_keys:
            print(f"[DEBUG] Cannot plot representation quality. Missing keys: {missing_keys}")
            print(f"[DEBUG] Available keys: {list(self.metrics_tracker.representation_metrics.keys())}")
            return
            
        # Create representation directory if not exists
        if 'representation' not in self.metrics_tracker.dirs:
            self.metrics_tracker.dirs['representation'] = os.path.join(self.metrics_tracker.results_dir, "representation_metrics")
            os.makedirs(self.metrics_tracker.dirs['representation'], exist_ok=True)
            
        # Determine plot directory and filename prefix
        if final:
            # Use current step count for final plots
            step_count = self.metrics_tracker.performance_metrics['total_env_steps']
            plot_dir = self.metrics_tracker.dirs['representation']
            filename_prefix = 'final_'
        else:
            # For intermediate plots, use milestone directory
            step_count = self.metrics_tracker.performance_metrics['total_env_steps']
            plot_dir = os.path.join(self.metrics_tracker.dirs['plots'], f'milestone_{step_count}')
            os.makedirs(plot_dir, exist_ok=True)
            filename_prefix = ''
        
        print(f"[DEBUG] Saving representation plots to {plot_dir}")
        
        # Get alignment data directly from metrics
        alignment_matrix = self.metrics_tracker.representation_metrics['alignment_matrix']
        best_matches = self.metrics_tracker.representation_metrics['best_matches']
        
        # Plot alignment heatmap
        print("[DEBUG] Plotting alignment heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(alignment_matrix, cmap='viridis', annot=True, fmt='.2f')
        plt.title('Learned Feature to Laplacian Eigenvector Alignment\n(Cosine Similarity)')
        plt.xlabel('Laplacian Eigenvector Index')
        plt.ylabel('Learned Feature Index')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{filename_prefix}eigenvector_alignment.png'))
        plt.close()
        
        # Sort matches by alignment value for better visualization
        best_matches.sort(key=lambda x: x[2], reverse=True)  # Sort by alignment value
        
        # Plot matching as a bar chart for top alignments
        print("[DEBUG] Plotting top alignments bar chart")
        plt.figure(figsize=(12, 8))
        num_to_show = min(10, len(best_matches))  # Show up to 10 matches
        feature_indices = [f"F{bm[0]}" for bm in best_matches[:num_to_show]]
        eigen_indices = [f"E{bm[1]}" for bm in best_matches[:num_to_show]]
        alignment_values = [bm[2] for bm in best_matches[:num_to_show]]
        
        # Create labels for x-axis: "Feature N -> Eigenvector M"
        x_labels = [f"{f} -> {e}" for f, e in zip(feature_indices, eigen_indices)]
        
        # Plot bar chart
        bars = plt.bar(range(len(x_labels)), alignment_values, color='skyblue')
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
        plt.ylim(0, 1.05)
        plt.title('Top Feature-Eigenvector Alignments')
        plt.ylabel('Cosine Similarity')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of each bar
        for bar, value in zip(bars, alignment_values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{filename_prefix}top_alignments.png'))
        plt.close()
        
        # Also save as text file for reference
        print("[DEBUG] Saving alignment data to text file")
        with open(os.path.join(plot_dir, f'{filename_prefix}eigenvector_alignment.txt'), 'w') as f:
            f.write("Feature-Eigenvector Alignment Results\n")
            f.write("===================================\n\n")
            f.write("Format: (Feature Index, Eigenvector Index, Cosine Similarity)\n\n")
            for i, (feat_idx, eigen_idx, similarity) in enumerate(best_matches):
                f.write(f"{i+1}. Feature {feat_idx} <-> Eigenvector {eigen_idx}: {similarity:.4f}\n")
                
        # Plot feature heatmaps on the maze grid
        print("[DEBUG] Plotting feature heatmaps")
        self._plot_feature_heatmaps(plot_dir, filename_prefix)
        
    def _plot_feature_heatmaps(self, plot_dir, filename_prefix):
        """Plot heatmaps of learned features across the maze grid."""
        # Create a subdirectory for feature heatmaps
        features_dir = os.path.join(plot_dir, 'feature_heatmaps')
        os.makedirs(features_dir, exist_ok=True)
        
        # Get necessary data
        if 'rep_matrix' not in self.metrics_tracker.representation_metrics or \
           'states' not in self.metrics_tracker.representation_metrics:
            print("[DEBUG] Missing representation data for heatmap plotting")
            return
            
        rep_matrix = self.metrics_tracker.representation_metrics['rep_matrix']
        states = self.metrics_tracker.representation_metrics['states']
        
        if len(states) == 0 or rep_matrix.shape[0] == 0:
            print("[DEBUG] No visited states available for heatmap plotting")
            return
        
        # Get maze dimensions
        maze_size = self.env.unwrapped.maze_size
        
        # For each feature dimension, create a heatmap on the maze grid
        num_features = min(10, rep_matrix.shape[1])  # Limit to first 10 features
        print(f"[DEBUG] Plotting {num_features} feature heatmaps for {len(states)} visited states")
        
        for feature_idx in range(num_features):
            # Create a grid to visualize this feature
            feature_grid = np.zeros((maze_size, maze_size))
            feature_grid.fill(np.nan)  # Fill with NaN for unvisited states
            
            # Fill in values for visited states
            for state_pos, rep_idx in zip(states, range(len(states))):
                if 0 <= state_pos[0] < maze_size and 0 <= state_pos[1] < maze_size:
                    feature_grid[state_pos] = rep_matrix[rep_idx, feature_idx]
            
            # Plot the feature heatmap
            plt.figure(figsize=(8, 6))
            
            # Use a diverging colormap centered at zero
            cmap = plt.cm.coolwarm
            masked_grid = np.ma.masked_invalid(feature_grid)  # Mask NaN values
            
            # If we have valid data, set limits based on data range
            if np.ma.count(masked_grid) > 0:
                # Calculate symmetric limits for better visualization
                abs_max = np.max(np.abs(masked_grid))
                vmin, vmax = -abs_max, abs_max
                plt.imshow(masked_grid, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
            else:
                # Just show empty grid if no valid data
                plt.imshow(masked_grid, cmap=cmap, interpolation='nearest')
                
            plt.colorbar(label=f'Feature {feature_idx} Value')
            plt.title(f'Feature {feature_idx} Representation Across State Space')
            plt.tight_layout()
            plt.savefig(os.path.join(features_dir, f'{filename_prefix}feature_{feature_idx}_heatmap.png'))
            plt.close()