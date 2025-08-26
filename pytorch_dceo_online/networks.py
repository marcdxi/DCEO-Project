"""
Neural network architectures for the fully online PyTorch implementation of Rainbow DCEO.
Following the implementation details in Klissarov et al., 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration without epsilon-greedy.
    
    Based on the paper: "Noisy Networks for Exploration" (Fortunato et al., 2018).
    """
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        """Forward pass with added noise."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
        
    def reset_parameters(self):
        """Reset trainable network parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset the noise for exploration."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        """Scale noise for factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class DuelingNetwork(nn.Module):
    """Dueling network architecture for value-based methods.
    
    Based on the paper: "Dueling Network Architectures for Deep Reinforcement Learning"
    (Wang et al., 2016).
    """
    
    def __init__(self, input_shape, num_actions, noisy=False, num_atoms=51):
        super(DuelingNetwork, self).__init__()
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.noisy = noisy
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        h = input_shape[1]  # Height
        w = input_shape[2]  # Width
        
        h = conv2d_size_out(h, 8, 4)
        h = conv2d_size_out(h, 4, 2)
        h = conv2d_size_out(h, 3, 1)
        
        w = conv2d_size_out(w, 8, 4)
        w = conv2d_size_out(w, 4, 2)
        w = conv2d_size_out(w, 3, 1)
        
        conv_output_size = 64 * h * w
        
        # Dueling architecture with value and advantage streams
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Value stream
        self.value_hidden = linear_layer(conv_output_size, 512)
        self.value_output = linear_layer(512, num_atoms)
        
        # Advantage stream
        self.advantage_hidden = linear_layer(conv_output_size, 512)
        self.advantage_output = linear_layer(512, num_actions * num_atoms)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)  # Flatten
        
        # Value stream
        value_hidden = F.relu(self.value_hidden(x))
        value = self.value_output(value_hidden).view(batch_size, 1, self.num_atoms)
        
        # Advantage stream
        advantage_hidden = F.relu(self.advantage_hidden(x))
        advantage = self.advantage_output(advantage_hidden).view(batch_size, self.num_actions, self.num_atoms)
        
        # Combine streams
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_atoms
    
    def reset_noise(self):
        """Reset noise for all noisy layers."""
        if not self.noisy:
            return
            
        self.value_hidden.reset_noise()
        self.value_output.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_output.reset_noise()


class FullRainbowNetwork(nn.Module):
    """Complete Rainbow network with distributional, dueling, and noisy features."""
    
    def __init__(self, input_shape, num_actions, noisy=True, dueling=True, distributional=True, num_atoms=51, v_min=-10, v_max=10):
        super(FullRainbowNetwork, self).__init__()
        
        self.num_actions = num_actions
        self.dueling = dueling
        self.distributional = distributional
        self.num_atoms = num_atoms
        self.noisy = noisy
        
        # Support for categorical distribution
        if distributional:
            self.register_buffer("supports", torch.linspace(v_min, v_max, num_atoms))
            self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Main network architecture
        if dueling:
            self.network = DuelingNetwork(input_shape, num_actions, noisy, num_atoms)
        else:
            # Non-dueling variant
            self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            
            # Calculate size after convolutions
            def conv2d_size_out(size, kernel_size, stride):
                return (size - (kernel_size - 1) - 1) // stride + 1
            
            h = input_shape[1]  # Height
            w = input_shape[2]  # Width
            
            h = conv2d_size_out(h, 8, 4)
            h = conv2d_size_out(h, 4, 2)
            h = conv2d_size_out(h, 3, 1)
            
            w = conv2d_size_out(w, 8, 4)
            w = conv2d_size_out(w, 4, 2)
            w = conv2d_size_out(w, 3, 1)
            
            conv_output_size = 64 * h * w
            
            # MLP head
            linear_layer = NoisyLinear if noisy else nn.Linear
            self.hidden = linear_layer(conv_output_size, 512)
            self.output = linear_layer(512, num_actions * num_atoms if distributional else num_actions)
    
    def forward(self, x, eval_mode=False):
        """Forward pass to calculate Q-values or distributional Q-values."""
        if self.dueling:
            q_atoms = self.network(x)
        else:
            batch_size = x.size(0)
            
            # Feature extraction
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(batch_size, -1)  # Flatten
            
            # MLP head
            x = F.relu(self.hidden(x))
            if self.distributional:
                q_atoms = self.output(x).view(batch_size, self.num_actions, self.num_atoms)
            else:
                q_atoms = self.output(x)
        
        if self.distributional:
            # Calculate Q-value expectations for argmax action selection
            q_values = (q_atoms * self.supports).sum(dim=2)
            
            # Get probabilities using softmax
            probabilities = F.softmax(q_atoms, dim=2)
            
            # Return both probabilities and Q-values in a dict
            return {"q_values": q_values, "probabilities": probabilities, "logits": q_atoms}
        else:
            # For non-distributional case, just return Q-values
            return {"q_values": q_atoms}
    
    def reset_noise(self):
        """Reset noise for all noisy layers."""
        if not self.noisy:
            return
            
        if self.dueling:
            self.network.reset_noise()
        else:
            if isinstance(self.hidden, NoisyLinear):
                self.hidden.reset_noise()
            if isinstance(self.output, NoisyLinear):
                self.output.reset_noise()


class LaplacianNetwork(nn.Module):
    """Laplacian representation network for DCEO.
    
    This network learns a latent representation that approximates the eigenvectors
    of the Laplacian of the state transition graph, as described in the 
    Klissarov et al. (2023) paper.
    """
    
    def __init__(self, input_shape, rep_dim=20, orthonormal=True):
        super(LaplacianNetwork, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        h = input_shape[1]  # Height
        w = input_shape[2]  # Width
        
        h = conv2d_size_out(h, 8, 4)
        h = conv2d_size_out(h, 4, 2)
        h = conv2d_size_out(h, 3, 1)
        
        w = conv2d_size_out(w, 8, 4)
        w = conv2d_size_out(w, 4, 2)
        w = conv2d_size_out(w, 3, 1)
        
        self.conv_output_size = 64 * h * w
        
        # Fully connected layers for representation
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc_rep = nn.Linear(512, rep_dim)
        
        self.rep_dim = rep_dim
        self.orthonormal = orthonormal
    
    def forward(self, x):
        """Forward pass to calculate representation vectors."""
        batch_size = x.size(0)
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)  # Flatten
        
        # Representation layers
        x = F.relu(self.fc1(x))
        rep = self.fc_rep(x)
        
        # Apply orthonormalization if needed (during training)
        if self.orthonormal and self.training and batch_size > 1:
            # Gram-Schmidt-like orthogonalization in batch
            rep = F.normalize(rep, p=2, dim=1)  # Normalize each representation vector
        
        return {"rep": rep, "features": x}
