"""
PyTorch implementation of the fully online Rainbow DCEO agent.
Following Algorithm 1 from Klissarov et al. (2023).
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from pytorch_dceo_online.networks import FullRainbowNetwork, LaplacianNetwork
from pytorch_dceo_online.replay_buffer import PrioritizedReplayBuffer

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FullyOnlineDCEOAgent:
    """Fully online implementation of the Rainbow DCEO agent.
    
    This implementation follows Algorithm 1 from Klissarov et al. (2023),
    where the representation learning, option discovery, and policy learning
    are all performed simultaneously in a unified process.
    """
    
    def __init__(self, 
                 input_shape, 
                 num_actions,
                 # Rainbow parameters
                 buffer_size=1000000,
                 batch_size=32,
                 gamma=0.99,
                 update_horizon=3,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 learning_rate=0.0000625,
                 noisy=True,
                 dueling=True,
                 double_dqn=True,
                 distributional=True,
                 num_atoms=51,
                 v_min=-10,
                 v_max=10,
                 # DCEO parameters
                 num_options=5,
                 option_prob=0.9,  # Probability of using options for exploration
                 option_duration=10,  # Average duration of an option
                 rep_dim=20,  # Dimension of the representation space
                 log_transform=True,  # Apply log transform to rewards
                 orthonormal=True,  # Apply orthonormalization to representations
                 alpha_rep=1.0,  # Coefficient for representation loss
                 alpha_main=1.0,  # Coefficient for main policy loss
                 alpha_option=1.0):  # Coefficient for option policy loss
        """Initialize the fully online Rainbow DCEO agent."""
        
        self.input_shape = input_shape  # (C, H, W) format
        self.num_actions = num_actions
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.cumulative_gamma = gamma ** update_horizon
        self.min_replay_history = min_replay_history
        self.update_period = update_period
        self.target_update_period = target_update_period
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.epsilon_decay_period = epsilon_decay_period
        self.eval_mode = False  # Training mode by default
        self.training_steps = 0
        
        # Rainbow features
        self.noisy = noisy
        self.dueling = dueling
        self.double_dqn = double_dqn
        self.distributional = distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # DCEO features
        self.num_options = num_options
        self.option_prob = option_prob
        self.dur = option_duration
        self.rep_dim = rep_dim
        self.orthonormal = orthonormal
        self.cur_opt = None
        self.option_term = True
        self.log_transform = log_transform
        
        # Loss coefficients
        self.alpha_rep = alpha_rep
        self.alpha_main = alpha_main
        self.alpha_option = alpha_option
        
        # Calculate supports for distributional RL
        self.supports = torch.linspace(v_min, v_max, num_atoms).to(device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Initialize networks
        
        # Main policy network for extrinsic rewards
        self.policy_net = FullRainbowNetwork(
            input_shape=input_shape,
            num_actions=num_actions,
            noisy=noisy,
            dueling=dueling,
            distributional=distributional,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max
        ).to(device)
        
        self.target_net = FullRainbowNetwork(
            input_shape=input_shape,
            num_actions=num_actions,
            noisy=noisy,
            dueling=dueling,
            distributional=distributional,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max
        ).to(device)
        
        # Copy parameters from policy to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is for evaluation only
        
        # Laplacian representation network
        self.rep_net = LaplacianNetwork(
            input_shape=input_shape,
            rep_dim=rep_dim,
            orthonormal=orthonormal
        ).to(device)
        
        # Option networks - one for each dimension of the representation
        self.option_nets = []
        self.option_targets = []
        for _ in range(num_options):
            option_net = FullRainbowNetwork(
                input_shape=input_shape,
                num_actions=num_actions,
                noisy=noisy,
                dueling=dueling,
                distributional=distributional,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max
            ).to(device)
            
            option_target = FullRainbowNetwork(
                input_shape=input_shape,
                num_actions=num_actions,
                noisy=noisy,
                dueling=dueling,
                distributional=distributional,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max
            ).to(device)
            
            option_target.load_state_dict(option_net.state_dict())
            option_target.eval()
            
            self.option_nets.append(option_net)
            self.option_targets.append(option_target)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, eps=0.00015)
        self.rep_optimizer = optim.Adam(self.rep_net.parameters(), lr=learning_rate, eps=0.00015)
        self.option_optimizers = [
            optim.Adam(option_net.parameters(), lr=learning_rate, eps=0.00015)
            for option_net in self.option_nets
        ]
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Action statistics for intrinsic reward weighting
        self.action_counts = np.zeros(num_actions)
        self.action_probs = np.ones(num_actions) / num_actions
        
        print(f"Fully Online Rainbow DCEO Agent initialized on {device}")
        print(f"Network architecture: Noisy={noisy}, Dueling={dueling}, Distributional={distributional}")
        print(f"DCEO parameters: Options={num_options}, Rep Dim={rep_dim}, Orthonormal={orthonormal}")
    
    def select_action(self, state, eval_mode=False):
        """Select an action using the fully online DCEO policy.
        
        This integrates option selection and termination in a unified process,
        allowing options to be used from the beginning of training.
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # When in evaluation mode, always use the greedy policy
        if eval_mode:
            with torch.no_grad():
                return self.policy_net(state)['q_values'].max(1)[1].item()
        
        # Option termination: Either continue with current option or terminate
        option_term = self.option_term or random.random() < (1 / self.dur)
        
        # Get epsilon for exploration
        epsilon = self._get_epsilon()
        
        # Exploration: Use either epsilon-greedy or an option
        if random.random() < epsilon:
            if self.noisy:
                # If using noisy networks, just reset the noise for exploration
                self.policy_net.reset_noise()
                for option_net in self.option_nets:
                    option_net.reset_noise()
                
                # Then decide whether to use an option or not
                if random.random() < self.option_prob and option_term and self.num_options > 0:
                    # Select a random option
                    self.cur_opt = random.randint(0, self.num_options - 1)
                    option_term = False
                    
                    # Use the option network to select action
                    with torch.no_grad():
                        action = self.option_nets[self.cur_opt](state)['q_values'].max(1)[1].item()
                else:
                    # Select a random action
                    action = random.randint(0, self.num_actions - 1)
            else:
                # When not using noisy networks, just do pure exploration with probability epsilon
                if random.random() < self.option_prob and option_term and self.num_options > 0:
                    # Select a random option
                    self.cur_opt = random.randint(0, self.num_options - 1)
                    option_term = False
                
                # Random action
                action = random.randint(0, self.num_actions - 1)
        else:
            # Exploitation: Use either the main policy or continue with an option
            with torch.no_grad():
                if self.cur_opt is not None and not option_term:
                    # Continue with the current option
                    action = self.option_nets[self.cur_opt](state)['q_values'].max(1)[1].item()
                else:
                    # Use the main policy
                    action = self.policy_net(state)['q_values'].max(1)[1].item()
        
        # Update option termination state
        self.option_term = option_term
        
        # Update action statistics
        self.action_counts[action] += 1
        total_count = np.sum(self.action_counts)
        self.action_probs = self.action_counts / total_count if total_count > 0 else np.ones(self.num_actions) / self.num_actions
        
        return action
    
    def _get_epsilon(self):
        """Calculate epsilon based on training progress."""
        if self.training_steps < self.min_replay_history:
            # Initial exploration phase
            return 1.0
        elif self.training_steps < self.min_replay_history + self.epsilon_decay_period:
            # Linear decay phase
            decay_steps = self.training_steps - self.min_replay_history
            return 1.0 - (1.0 - self.epsilon_train) * decay_steps / self.epsilon_decay_period
        else:
            # Final epsilon value
            return self.epsilon_train
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(
            state=np.array(state, dtype=np.float32),
            action=action,
            reward=reward,
            next_state=np.array(next_state, dtype=np.float32) if not done else np.zeros_like(state, dtype=np.float32),
            done=float(done)
        )
    
    def update(self):
        """Perform a fully online update of all networks simultaneously.
        
        This is the core of the fully online algorithm, where representation learning,
        option discovery, and policy learning all happen in an integrated process.
        """
        self.training_steps += 1
        
        # Only update after min_replay_history and on update_period
        if (self.training_steps < self.min_replay_history or
            self.training_steps % self.update_period != 0):
            return
        
        # Sample a batch from replay buffer - this batch will be used for all updates
        if len(self.replay_buffer) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size, device)
        
        # -----------------------------------
        # 1. Update Laplacian representation
        # -----------------------------------
        rep_dict = self.rep_net(states)
        next_rep_dict = self.rep_net(next_states)
        
        rep = rep_dict['rep']
        next_rep = next_rep_dict['rep']
        
        # Compute the Laplacian representation loss
        rep_loss = self._compute_laplacian_loss(
            rep, next_rep, rewards, dones, actions
        )
        
        # ------------------------------
        # 2. Update main policy network
        # ------------------------------
        main_loss, main_td_errors = self._compute_policy_loss(
            self.policy_net, self.target_net, 
            states, actions, rewards, next_states, dones, weights
        )
        
        # ----------------------------
        # 3. Update option networks
        # ----------------------------
        option_losses = []
        option_td_errors = []
        
        for i in range(self.num_options):
            # Compute intrinsic rewards based on the representation
            intrinsic_rewards = self._compute_intrinsic_rewards(
                rep, next_rep, i, actions
            )
            
            # Update the option policy with intrinsic rewards
            opt_loss, opt_td = self._compute_policy_loss(
                self.option_nets[i], self.option_targets[i],
                states, actions, intrinsic_rewards, next_states, dones, weights
            )
            
            option_losses.append(opt_loss)
            option_td_errors.append(opt_td)
        
        # ------------------------------------------
        # 4. Combine losses and perform optimization
        # ------------------------------------------
        
        # Combine TD errors for prioritized replay
        # Use a weighted combination of main policy and option TD errors
        combined_td_errors = main_td_errors.clone()
        for i in range(self.num_options):
            combined_td_errors = torch.max(combined_td_errors, option_td_errors[i])
        
        # Update priorities in replay buffer
        new_priorities = combined_td_errors.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)
        
        # Apply all updates in a fully online manner
        
        # Update representation network
        self.rep_optimizer.zero_grad()
        if rep_loss.requires_grad:
            (self.alpha_rep * rep_loss).backward(retain_graph=True)  # Retain graph for subsequent backward calls
            torch.nn.utils.clip_grad_norm_(self.rep_net.parameters(), 10.0)
            self.rep_optimizer.step()
        
        # Update main policy network
        self.policy_optimizer.zero_grad()
        if main_loss.requires_grad:
            (self.alpha_main * main_loss).backward(retain_graph=True)  # Retain graph for option losses
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.policy_optimizer.step()
        
        # Update option networks
        for i in range(self.num_options):
            self.option_optimizers[i].zero_grad()
            if option_losses[i].requires_grad:
                # Last option doesn't need to retain graph
                retain = (i < self.num_options - 1)
                (self.alpha_option * option_losses[i]).backward(retain_graph=retain)
                torch.nn.utils.clip_grad_norm_(self.option_nets[i].parameters(), 10.0)
                self.option_optimizers[i].step()
        
        # Reset noise for all networks if using noisy networks
        if self.noisy:
            self.policy_net.reset_noise()
            for opt_net in self.option_nets:
                opt_net.reset_noise()
        
        # Periodically update target networks
        if self.training_steps % self.target_update_period == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            for i in range(self.num_options):
                self.option_targets[i].load_state_dict(self.option_nets[i].state_dict())
    
    def _compute_laplacian_loss(self, rep, next_rep, rewards, dones, actions):
        """Compute the loss for the Laplacian representation.
        
        This implements the representation learning objective from the paper,
        which approximates the eigenvectors of the Laplacian of the state
        transition graph.
        """
        batch_size = rep.size(0)
        
        # Prepare action encoding (one-hot) for incorporating action statistics
        action_one_hot = F.one_hot(actions, self.num_actions).float()
        
        # Compute log probabilities of actions for inverse frequency weighting
        log_action_probs = torch.FloatTensor(-np.log(self.action_probs + 1e-6)).to(device)
        
        # Get weights for each action in the batch
        action_weights = torch.sum(action_one_hot * log_action_probs.unsqueeze(0), dim=1)
        
        # Apply optional log transform to rewards
        if self.log_transform:
            reward_signs = torch.sign(rewards)
            reward_vals = torch.log(torch.abs(rewards) + 1.0)
            weighted_rewards = reward_signs * reward_vals
        else:
            weighted_rewards = rewards
        
        # Scale rewards by action rarity
        weighted_rewards = weighted_rewards * action_weights
        
        # Compute temporal difference loss for representation
        # rep_target = rep + γ * next_rep - rep (if not done)
        rep_target = rep.clone().detach()
        rep_target = rep_target + weighted_rewards.unsqueeze(1).repeat(1, self.rep_dim) * \
                   (1.0 - dones.unsqueeze(1).repeat(1, self.rep_dim)) * \
                   self.gamma * next_rep - rep
        
        # MSE loss for representation learning
        rep_loss = F.mse_loss(rep, rep_target)
        
        # Add orthonormality constraint if enabled
        if self.orthonormal and batch_size > 1:
            # Compute the Gram matrix of the representations
            gram = torch.mm(rep.t(), rep)
            
            # Target is the identity matrix (orthonormal vectors)
            identity = torch.eye(self.rep_dim).to(device)
            
            # Loss is the distance from the Gram matrix to the identity
            ortho_loss = F.mse_loss(gram, identity)
            
            # Combine losses
            rep_loss = rep_loss + 0.1 * ortho_loss
        
        return rep_loss
    
    def _compute_intrinsic_rewards(self, rep, next_rep, option_idx, actions):
        """Compute intrinsic rewards for option learning based on representation.
        
        This generates rewards based on the change in the representation along
        a specific dimension, encouraging the option to maximize this change.
        """
        if option_idx >= self.rep_dim:
            return torch.zeros_like(actions, dtype=torch.float)
        
        # Extract the specific representation dimension for this option
        rep_dim = rep[:, option_idx]
        next_rep_dim = next_rep[:, option_idx]
        
        # Compute the change in representation
        rep_change = next_rep_dim - rep_dim
        
        # Scale by action rarity
        action_one_hot = F.one_hot(actions, self.num_actions).float()
        log_action_probs = torch.FloatTensor(-np.log(self.action_probs + 1e-6)).to(device)
        action_weights = torch.sum(action_one_hot * log_action_probs.unsqueeze(0), dim=1)
        
        # Final intrinsic reward
        intrinsic_rewards = rep_change * action_weights
        
        return intrinsic_rewards
    
    def _compute_policy_loss(self, policy_net, target_net, states, actions, rewards, next_states, dones, weights):
        """Compute the loss for a policy network.
        
        This handles both distributional and non-distributional cases,
        as well as double Q-learning if enabled.
        """
        # Calculate current Q-values
        if self.distributional:
            current_q_logits = policy_net(states)['logits']
            batch_size = current_q_logits.size(0)
            current_q_dist = F.softmax(current_q_logits, dim=2)
            current_q_action_dist = current_q_dist[torch.arange(batch_size), actions]
            
            # Use action values for training priorities
            current_q = (current_q_action_dist * self.supports).sum(1)
        else:
            current_q = policy_net(states)['q_values'][torch.arange(states.size(0)), actions]
        
        # Calculate target Q-values
        with torch.no_grad():
            if self.distributional:
                if self.double_dqn:
                    # Double DQN: select actions using policy network
                    next_q_values = policy_net(next_states)['q_values']
                    next_actions = next_q_values.argmax(1)
                else:
                    # Regular DQN: select actions using target network
                    next_q_values = target_net(next_states)['q_values']
                    next_actions = next_q_values.argmax(1)
                
                # Get next state distributional values for selected actions
                next_q_logits = target_net(next_states)['logits']
                batch_size = next_q_logits.size(0)
                next_q_dist = F.softmax(next_q_logits, dim=2)
                next_q_action_dist = next_q_dist[torch.arange(batch_size), next_actions]
                
                # Calculate projected distributional target
                # Compute projection of the distribution onto the support
                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)
                supports = self.supports.unsqueeze(0)
                
                # Project the distribution onto the support
                tz = rewards + (1.0 - dones) * self.cumulative_gamma * supports
                tz = tz.clamp(min=self.v_min, max=self.v_max)
                
                # Get probabilities on the fixed support
                b = (tz - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()
                
                # Fix disappearing probability mass when l = b = u (b is int)
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.num_atoms - 1)) * (l == u)] += 1
                
                # Distribute probability mass
                batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, self.num_atoms)
                
                target_dist = torch.zeros(batch_size, self.num_atoms).to(device)
                
                # Distribute probability to lower atom
                target_dist.scatter_add_(
                    1, l, next_q_action_dist * (u.float() - b)
                )
                
                # Distribute probability to upper atom
                target_dist.scatter_add_(
                    1, u, next_q_action_dist * (b - l.float())
                )
                
                # Calculate KL divergence loss
                log_probs = F.log_softmax(current_q_logits[torch.arange(batch_size), actions], dim=1)
                loss = -(target_dist * log_probs).sum(1)
                
                # Scale by IS weights for prioritized replay
                loss = (loss * weights).mean()
                
                # Calculate TD errors for replay buffer priorities
                td_error = torch.abs(current_q - (target_dist * self.supports).sum(1))
            else:
                # Non-distributional DQN
                if self.double_dqn:
                    # Double DQN: select actions using policy network
                    next_actions = policy_net(next_states)['q_values'].argmax(1)
                    next_q = target_net(next_states)['q_values'][torch.arange(batch_size), next_actions]
                else:
                    # Regular DQN: use max Q-value from target network
                    next_q = target_net(next_states)['q_values'].max(1)[0]
                
                # Calculate target values
                target_q = rewards + (1.0 - dones) * self.cumulative_gamma * next_q
                
                # MSE loss
                loss = F.mse_loss(current_q, target_q, reduction='none')
                
                # Scale by IS weights for prioritized replay
                loss = (loss * weights).mean()
                
                # Calculate TD errors for replay buffer priorities
                td_error = torch.abs(current_q - target_q)
        
        return loss, td_error.detach()
    
    def save(self, save_dir='./checkpoints'):
        """Save the agent's parameters with robust error handling."""
        import os
        import tempfile
        
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Use a temporary file approach for safer saving
        try:
            # Save main policy and representation networks
            main_path = os.path.join(save_dir, 'dceo_online_main.pt')
            temp_main = None
            
            # Create a temporary file in the same directory
            with tempfile.NamedTemporaryFile(delete=False, dir=save_dir, suffix='.pt') as tmp:
                temp_main = tmp.name
                
            # Save to the temporary file first
            torch.save({
                'policy_state_dict': self.policy_net.state_dict(),
                'target_state_dict': self.target_net.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'rep_net_state_dict': self.rep_net.state_dict(),
                'rep_optimizer_state_dict': self.rep_optimizer.state_dict(),
                'training_steps': self.training_steps,
                'action_counts': self.action_counts,
            }, temp_main)
            
            # If successful, rename the temporary file to the target path
            if os.path.exists(main_path):
                os.remove(main_path)  # Remove existing file if it exists
            os.rename(temp_main, main_path)
            print(f"Main model saved to {main_path}")
            
            # Save option networks individually to avoid large single file
            for i in range(self.num_options):
                option_path = os.path.join(save_dir, f'dceo_online_option_{i}.pt')
                temp_option = None
                
                with tempfile.NamedTemporaryFile(delete=False, dir=save_dir, suffix='.pt') as tmp:
                    temp_option = tmp.name
                
                torch.save({
                    'option_net_state_dict': self.option_nets[i].state_dict(),
                    'option_target_state_dict': self.option_targets[i].state_dict(),
                    'option_optimizer_state_dict': self.option_optimizers[i].state_dict(),
                }, temp_option)
                
                if os.path.exists(option_path):
                    os.remove(option_path)
                os.rename(temp_option, option_path)
            
            print(f"All option models saved to {save_dir}")
            return True
            
        except Exception as e:
            print(f"Error during save operation: {str(e)}")
            print("Attempting fallback save method...")
            
            # Fallback: Save components to separate smaller files
            try:
                backup_dir = os.path.join(save_dir, 'backup_parts')
                os.makedirs(backup_dir, exist_ok=True)
                
                # Save policy networks
                torch.save(self.policy_net.state_dict(), os.path.join(backup_dir, 'policy_net.pt'))
                torch.save(self.target_net.state_dict(), os.path.join(backup_dir, 'target_net.pt'))
                torch.save(self.policy_optimizer.state_dict(), os.path.join(backup_dir, 'policy_optimizer.pt'))
                
                # Save representation network
                torch.save(self.rep_net.state_dict(), os.path.join(backup_dir, 'rep_net.pt'))
                torch.save(self.rep_optimizer.state_dict(), os.path.join(backup_dir, 'rep_optimizer.pt'))
                
                # Save metadata
                import numpy as np
                np.save(os.path.join(backup_dir, 'training_steps.npy'), self.training_steps)
                np.save(os.path.join(backup_dir, 'action_counts.npy'), self.action_counts)
                
                # Save options separately
                for i in range(self.num_options):
                    torch.save(self.option_nets[i].state_dict(), os.path.join(backup_dir, f'option_net_{i}.pt'))
                    torch.save(self.option_targets[i].state_dict(), os.path.join(backup_dir, f'option_target_{i}.pt'))
                    torch.save(self.option_optimizers[i].state_dict(), os.path.join(backup_dir, f'option_optimizer_{i}.pt'))
                
                print(f"Fallback save completed to {backup_dir}")
                return True
            except Exception as backup_e:
                print(f"Both save methods failed. Error: {str(backup_e)}")
                return False
    
    def load(self, save_dir='./checkpoints'):
        """Load the agent's parameters."""
        import os
        
        # Load main networks if file exists
        main_path = os.path.join(save_dir, 'dceo_online_main.pt')
        if os.path.exists(main_path):
            checkpoint = torch.load(main_path)
            
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.rep_net.load_state_dict(checkpoint['rep_net_state_dict'])
            self.rep_optimizer.load_state_dict(checkpoint['rep_optimizer_state_dict'])
            self.training_steps = checkpoint['training_steps']
            self.action_counts = checkpoint['action_counts']
            
            print(f"Loaded main agent from {main_path} (step {self.training_steps})")
        
        # Load option networks if files exist
        for i in range(self.num_options):
            option_path = os.path.join(save_dir, f'dceo_online_option_{i}.pt')
            if os.path.exists(option_path):
                checkpoint = torch.load(option_path)
                
                self.option_nets[i].load_state_dict(checkpoint['option_net_state_dict'])
                self.option_targets[i].load_state_dict(checkpoint['option_target_state_dict'])
                self.option_optimizers[i].load_state_dict(checkpoint['option_optimizer_state_dict'])
                
                print(f"Loaded option {i} from {option_path}")
    
    def reset_exploration(self):
        """Reset the exploration parameters."""
        self.training_steps = 0
        self.action_counts = np.zeros(self.num_actions)
        self.action_probs = np.ones(self.num_actions) / self.num_actions
        self.cur_opt = None
        self.option_term = True
