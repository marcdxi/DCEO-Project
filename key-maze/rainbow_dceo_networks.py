"""
Network architectures for Rainbow DCEO agent.
This implementation uses PyTorch for better compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """Implementation of Noisy Networks for Exploration (NoisyNet)."""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Mean weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """Scale noise by signed square root transformation."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """Reset the noise parameters."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """Forward pass with noise injection."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DQNNetwork(nn.Module):
    """Basic DQN network architecture."""
    
    def __init__(self, input_shape, num_actions, noisy=False, dueling=False):
        super(DQNNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        self.dueling = dueling
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the feature map after convolutions
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Value and advantage streams for dueling architecture
        if noisy:
            if dueling:
                self.value_stream = nn.Sequential(
                    NoisyLinear(conv_output_size, 512),
                    nn.ReLU(),
                    NoisyLinear(512, 1)
                )
                self.advantage_stream = nn.Sequential(
                    NoisyLinear(conv_output_size, 512),
                    nn.ReLU(),
                    NoisyLinear(512, num_actions)
                )
            else:
                self.fc = nn.Sequential(
                    NoisyLinear(conv_output_size, 512),
                    nn.ReLU(),
                    NoisyLinear(512, num_actions)
                )
        else:
            if dueling:
                self.value_stream = nn.Sequential(
                    nn.Linear(conv_output_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1)
                )
                self.advantage_stream = nn.Sequential(
                    nn.Linear(conv_output_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_actions)
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(conv_output_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_actions)
                )
    
    def _get_conv_output_size(self, shape):
        """Calculate the size of the flattened convolutional output."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()))
    
    def reset_noise(self):
        """Reset noise for all noisy layers."""
        if not self.noisy:
            return
        
        if self.dueling:
            for module in self.value_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
            for module in self.advantage_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
        else:
            for module in self.fc:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
    def forward(self, x):
        """Forward pass through the network."""
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dueling architecture
        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.fc(x)
        
        return q_values


class RainbowNetwork(nn.Module):
    """Rainbow DQN network architecture with distributional RL."""
    
    def __init__(self, input_shape, num_actions, num_atoms=51, vmin=-10, vmax=10, 
                 noisy=True, dueling=True):
        super(RainbowNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.noisy = noisy
        self.dueling = dueling
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the feature map after convolutions
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Value and advantage streams for dueling architecture
        if noisy:
            if dueling:
                self.value_stream = nn.Sequential(
                    NoisyLinear(conv_output_size, 512),
                    nn.ReLU(),
                    NoisyLinear(512, num_atoms)
                )
                self.advantage_stream = nn.Sequential(
                    NoisyLinear(conv_output_size, 512),
                    nn.ReLU(),
                    NoisyLinear(512, num_actions * num_atoms)
                )
            else:
                self.fc = nn.Sequential(
                    NoisyLinear(conv_output_size, 512),
                    nn.ReLU(),
                    NoisyLinear(512, num_actions * num_atoms)
                )
        else:
            if dueling:
                self.value_stream = nn.Sequential(
                    nn.Linear(conv_output_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_atoms)
                )
                self.advantage_stream = nn.Sequential(
                    nn.Linear(conv_output_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_actions * num_atoms)
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(conv_output_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_actions * num_atoms)
                )
        
        self.register_buffer('supports', torch.linspace(vmin, vmax, num_atoms))
        self.delta_z = (vmax - vmin) / (num_atoms - 1)
    
    def _get_conv_output_size(self, shape):
        """Calculate the size of the flattened convolutional output."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()))
    
    def reset_noise(self):
        """Reset noise for all noisy layers."""
        if not self.noisy:
            return
        
        if self.dueling:
            for module in self.value_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
            for module in self.advantage_stream:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
        else:
            for module in self.fc:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
    def forward(self, x):
        """Forward pass through the network."""
        batch_size = x.size(0)
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Distributional RL with dueling architecture
        if self.dueling:
            value = self.value_stream(x).view(batch_size, 1, self.num_atoms)
            advantage = self.advantage_stream(x).view(batch_size, self.num_actions, self.num_atoms)
            q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_atoms = self.fc(x).view(batch_size, self.num_actions, self.num_atoms)
        
        # Apply softmax to convert to probabilities
        dist = F.softmax(q_atoms, dim=2)
        # Calculate expected values
        q = torch.sum(dist * self.supports, dim=2)
        
        return q, dist


class RepresentationNetwork(nn.Module):
    """Network for learning Laplacian representation."""
    
    def __init__(self, input_shape, rep_dim=10):
        super(RepresentationNetwork, self).__init__()
        self.input_shape = input_shape
        self.rep_dim = rep_dim
        
        # Feature extraction layers (same as DQN)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the feature map after convolutions
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Representation output
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, rep_dim)
        )
    
    def _get_conv_output_size(self, shape):
        """Calculate the size of the flattened convolutional output."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()))
    
    def forward(self, x):
        """Forward pass through the network."""
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Representation
        phi = self.fc(x)
        
        return phi
