"""
Mixed precision utilities for reinforcement learning agents.
"""
import torch
from torch.cuda.amp import autocast, GradScaler

# Create a global scaler to be used across all agents
scaler = GradScaler() if torch.cuda.is_available() else None

def is_amp_available():
    """Check if mixed precision is available."""
    return torch.cuda.is_available()

def wrap_ddqn_learn(agent):
    """
    Wrap DDQN agent's _learn method with mixed precision.
    """
    if not is_amp_available() or not hasattr(agent, '_learn'):
        return  # No change if AMP not available or _learn not present
    
    # Store original method
    if not hasattr(agent, '_original_learn'):
        agent._original_learn = agent._learn
        
        # Create mixed precision version
        def amp_learn(self, experiences=None, batch_size=None):
            if experiences is None:
                experiences = self.replay_buffer.sample(batch_size or self.batch_size)
                
            states, actions, rewards, next_states, dones = experiences
            
            # Forward pass with autocast for mixed precision
            with autocast():
                # Current Q values
                q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Next Q values with target network
                with torch.no_grad():
                    # Double DQN
                    next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
                    next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                    
                # Calculate target
                targets = rewards + self.gamma * next_q_values * (1 - dones)
                
                # Compute loss
                loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
            
            return loss.item()
            
        # Replace the method
        agent._learn = type(agent)._learn.__get__(amp_learn, type(agent))
        print("DDQN agent's learning method enhanced with mixed precision")

def wrap_rnd_learn(agent):
    """
    Wrap RND agent's _learn method with mixed precision.
    """
    if not is_amp_available() or not hasattr(agent, '_learn'):
        return  # No change if AMP not available or _learn not present
    
    # Store original methods
    if not hasattr(agent, '_original_learn'):
        agent._original_learn = agent._learn
        agent._original_update_q_network = agent._update_q_network
        agent._original_update_rnd_predictor = agent._update_rnd_predictor
        
        # Create mixed precision version for Q-network update
        def amp_update_q_network(self, states, actions, rewards, next_states, dones):
            with autocast():
                # Get current Q values
                q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute next state values using target network
                with torch.no_grad():
                    # Double Q-learning
                    next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
                    next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                    
                # Compute TD targets
                targets = rewards + self.gamma * next_q_values * (1 - dones)
                
                # Compute loss
                q_loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
            
            # Backward pass with scaling
            scaler.scale(q_loss).backward()
            scaler.step(self.q_optimizer)
            scaler.update()
            self.q_optimizer.zero_grad()
            
            return q_loss.item()
            
        # Create mixed precision version for RND predictor update
        def amp_update_rnd_predictor(self, states):
            with autocast():
                # Get target and predicted features
                with torch.no_grad():
                    target_features = self.rnd_target(states)
                predictor_features = self.rnd_predictor(states)
                
                # Compute intrinsic loss (mean squared error)
                rnd_loss = torch.nn.functional.mse_loss(predictor_features, target_features)
            
            # Backward pass with scaling
            scaler.scale(rnd_loss).backward()
            scaler.step(self.rnd_optimizer)
            scaler.update()
            self.rnd_optimizer.zero_grad()
            
            return rnd_loss.item()
            
        # Create mixed precision version of main learn method
        def amp_learn(self):
            if len(self.memory) < self.batch_size:
                return
                
            # Sample experiences from replay buffer
            experiences = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = experiences
            
            # Update networks with mixed precision
            q_loss = self._update_q_network(states, actions, rewards, next_states, dones)
            rnd_loss = self._update_rnd_predictor(states)
            
            return q_loss, rnd_loss
            
        # Replace the methods
        agent._update_q_network = type(agent)._update_q_network.__get__(amp_update_q_network, type(agent))
        agent._update_rnd_predictor = type(agent)._update_rnd_predictor.__get__(amp_update_rnd_predictor, type(agent))
        agent._learn = type(agent)._learn.__get__(amp_learn, type(agent))
        print("RND agent's learning methods enhanced with mixed precision")
