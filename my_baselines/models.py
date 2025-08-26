"""
Neural network models for baseline reinforcement learning algorithms.
Compatible with maze environments from the DCEO project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """
    Basic Q-Network for discrete action spaces.
    Uses a convolutional architecture for image-based inputs.
    Dynamically handles different input shapes (e.g., Complex Maze with 3 channels vs Key Maze with 5 channels)
    """
    def __init__(self, input_shape, num_actions):
        """
        Initialize a dueling Q-network for value approximation.
        
        Args:
            input_shape (tuple): The input shape (channels, height, width) or (height, width, channels)
            num_actions (int): The number of possible actions
        """
        super(QNetwork, self).__init__()
        
        # Handle different input shape formats (CHW vs HWC)
        if len(input_shape) == 3:
            if input_shape[0] <= 5:  # If first dimension is small, assume CHW format
                self.input_shape = input_shape
            else:  # Otherwise, assume HWC format and convert to CHW
                self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        else:
            # Default for unexpected shapes
            self.input_shape = input_shape
            
        self.num_actions = num_actions
        print(f"QNetwork initialized with input shape: {self.input_shape}")
        # First dimension of input_shape is number of channels
        
        # Feature extraction layers - first dimension is # of channels
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the features after convolutions
        # This will depend on the input shape (especially the number of channels)
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Value stream (for DDQN)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream (for DDQN)
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def _get_conv_output_size(self, shape):
        """Calculate the size of the flattened features after convolution layers."""
        # Create a dummy tensor with proper shape and format (batch, channels, height, width)
        if len(shape) == 3:
            # Handle the case where shape is passed as (channels, height, width) or (height, width, channels)
            if shape[0] <= 5:  # First dimension is probably channels (CHW format)
                dummy_input = torch.zeros(1, *shape)
            else:  # First dimension is probably height (HWC format), convert to CHW
                dummy_input = torch.zeros(1, shape[2], shape[0], shape[1])
        else:
            # Default case
            dummy_input = torch.zeros(1, *shape)
        
        # Run through the network
        o = self._forward_conv(dummy_input)
        return int(np.prod(o.size()))
        
    def _forward_conv(self, x):
        """Forward pass through convolutional layers."""
        x = self.conv1(x)
        x = F.relu(x)
        # Adjust max_pool for smaller mazes - use pooling with size 2 but stride 1 to avoid dimension reduction
        if min(x.shape[2], x.shape[3]) <= 12:  # For small mazes
            x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0)
        else:
            x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        # Second pooling layer adjustment
        if min(x.shape[2], x.shape[3]) <= 8:  # For small mazes after first pooling
            x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0)
        else:
            x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        return x
    
    def forward(self, x):
        """
        Forward pass through the dueling Q-network.
        
        Args:
            x: Input tensor of shape (batch_size, C, H, W)
            
        Returns:
            Q-values for each action
        """
        # Make sure input is the right format
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Run through network
        conv_out = self._forward_conv(x)
        flattened = conv_out.reshape(conv_out.size(0), -1)
        
        # Handle dynamic dimensions for value and advantage streams
        flattened_size = flattened.size(1)
        expected_size = 1024  # Size expected during initialization
        
        if flattened_size != expected_size:
            # Create dynamic value and advantage streams if needed
            if not hasattr(self, '_dynamic_value_stream') or self._dynamic_value_stream[0].in_features != flattened_size:
                print(f"INFO: QNetwork adapting to input size {flattened_size} (one-time message)")
                self._dynamic_value_stream = nn.Sequential(
                    nn.Linear(flattened_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                ).to(flattened.device)
                
                self._dynamic_advantage_stream = nn.Sequential(
                    nn.Linear(flattened_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.num_actions)
                ).to(flattened.device)
            
            # Use dynamic streams
            value = self._dynamic_value_stream(flattened)
            advantage = self._dynamic_advantage_stream(flattened)
        else:
            # Use original streams
            value = self.value_stream(flattened)
            advantage = self.advantage_stream(flattened)
        
        # Combine streams Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class RNDPredictor(nn.Module):
    """
    Predictor network for RND.
    Tries to predict the output of a fixed randomly initialized target network.
    Dynamically handles different input shapes (Complex Maze vs Key Maze).
    """
    def __init__(self, input_shape, output_dim=512):
        """
        Initialize the RND predictor network.
        
        Args:
            input_shape: Shape of the input observation (C, H, W)
            output_dim: Output dimension
        """
        super(RNDPredictor, self).__init__()
        
        # Handle different input shape formats (CHW vs HWC)
        if len(input_shape) == 3:
            if input_shape[0] <= 5:  # If first dimension is small, assume CHW format
                self.input_shape = input_shape
            else:  # Otherwise, assume HWC format and convert to CHW
                self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        else:
            # Default for unexpected shapes
            self.input_shape = input_shape
            
        self.output_dim = output_dim
        print(f"RNDPredictor initialized with input shape: {self.input_shape}")
        
        # Feature extractor individual layers for more control
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Predictor layers with dynamic size based on conv output
        print(f"RNDPredictor using conv_output_size: {conv_output_size}")
        self.predictor = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def _get_conv_output_size(self, shape):
        """Calculate the size of the flattened features after convolution layers."""
        o = self._forward_features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def _forward_features(self, x):
        """Forward pass through feature extractor."""
        if len(x.shape) == 3:  # Add batch dimension if missing
            x = x.unsqueeze(0)
        # Ensure proper channel dimension - rearrange if needed
        if x.shape[1] != self.input_shape[0]:  # If channels are not in second dimension
            # Assume batch, height, width, channels format, convert to batch, channels, height, width
            if len(x.shape) == 4 and x.shape[3] == self.input_shape[0]:
                x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        # Forward through convolutional layers with adaptive pooling
        x = self.conv1(x)
        x = F.relu(x)
        
        # Adjust max_pool for smaller mazes
        if min(x.shape[2], x.shape[3]) <= 12:  # For small mazes
            x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0)
        else:
            x = F.max_pool2d(x, 2)
            
        x = self.conv2(x)
        x = F.relu(x)
        
        # Second pooling layer adjustment
        if min(x.shape[2], x.shape[3]) <= 8:  # For small mazes after first pooling
            x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0)
        else:
            x = F.max_pool2d(x, 2)
            
        x = self.conv3(x)
        x = F.relu(x)
        
        return x
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Ensure input has batch dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Pass through feature extraction layers
        features = self._forward_features(x)
        # Use reshape instead of view to handle non-contiguous tensors
        flattened = features.reshape(x.size(0), -1)
        
        # Check tensor dimensions
        flattened_size = flattened.size(1)
        if flattened_size != 1024:  # If size doesn't match expected
            # Create a new linear layer with the correct input size if needed
            if not hasattr(self, '_dynamic_predictor') or self._dynamic_predictor[0].in_features != flattened_size:
                print(f"INFO: RNDPredictor adapting to input size {flattened_size} (one-time message)")
                self._dynamic_predictor = nn.Sequential(
                    nn.Linear(flattened_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.output_dim)
                ).to(flattened.device)
            return self._dynamic_predictor(flattened)
        
        # Use original predictor if sizes match
        return self.predictor(flattened)


class RNDTarget(nn.Module):
    """
    Fixed random target network for RND.
    Used as a fixed random embedding of observations.
    Dynamically handles different input shapes (Complex Maze vs Key Maze).
    """
    def __init__(self, input_shape, output_dim=512):
        """
        Initialize the RND target network.
        
        Args:
            input_shape: Shape of the input observation (C, H, W)
            output_dim: Dimension of the output embedding
        """
        super(RNDTarget, self).__init__()
        
        # Handle different input shape formats (CHW vs HWC)
        if len(input_shape) == 3:
            if input_shape[0] <= 5:  # If first dimension is small, assume CHW format
                self.input_shape = input_shape
            else:  # Otherwise, assume HWC format and convert to CHW
                self.input_shape = (input_shape[2], input_shape[0], input_shape[1])
        else:
            # Default for unexpected shapes
            self.input_shape = input_shape
            
        self.output_dim = output_dim
        print(f"RNDTarget initialized with input shape: {self.input_shape}")
        
        # Feature extractor individual layers for more control
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Target network layers with dynamic size based on conv output
        print(f"RNDTarget using conv_output_size: {conv_output_size}")
        self.target_network = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        # Initialize with random weights and freeze
        self._init_weights()
        self._freeze_weights()
        
    def _init_weights(self):
        """Initialize weights with random values."""
        for param in self.parameters():
            param.data.normal_(0.0, 0.1)
            
    def _freeze_weights(self):
        """Freeze all weights so they don't update during training."""
        for param in self.parameters():
            param.requires_grad = False
            
    def _get_conv_output_size(self, shape):
        """Calculate the size of the flattened features after convolution layers."""
        o = self._forward_features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def _forward_features(self, x):
        """Forward pass through feature extractor."""
        if len(x.shape) == 3:  # Add batch dimension if missing
            x = x.unsqueeze(0)
        # Ensure proper channel dimension - rearrange if needed
        if x.shape[1] != self.input_shape[0]:  # If channels are not in second dimension
            # Assume batch, height, width, channels format, convert to batch, channels, height, width
            if len(x.shape) == 4 and x.shape[3] == self.input_shape[0]:
                x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        # Forward through convolutional layers with adaptive pooling
        x = self.conv1(x)
        x = F.relu(x)
        
        # Adjust max_pool for smaller mazes
        if min(x.shape[2], x.shape[3]) <= 12:  # For small mazes
            x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0)
        else:
            x = F.max_pool2d(x, 2)
            
        x = self.conv2(x)
        x = F.relu(x)
        
        # Second pooling layer adjustment
        if min(x.shape[2], x.shape[3]) <= 8:  # For small mazes after first pooling
            x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0)
        else:
            x = F.max_pool2d(x, 2)
            
        x = self.conv3(x)
        x = F.relu(x)
        
        return x
        
    def forward(self, x):
        """
        Forward pass through the target network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Ensure input has batch dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Pass through feature extraction layers
        features = self._forward_features(x)
        # Use reshape instead of view to handle non-contiguous tensors
        flattened = features.reshape(x.size(0), -1)
        
        # Check tensor dimensions
        flattened_size = flattened.size(1)
        if flattened_size != 1024:  # If size doesn't match expected
            # Create a new linear layer with the correct input size if needed
            if not hasattr(self, '_dynamic_target') or self._dynamic_target[0].in_features != flattened_size:
                print(f"INFO: RNDTarget adapting to input size {flattened_size} (one-time message)")
                self._dynamic_target = nn.Sequential(
                    nn.Linear(flattened_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, self.output_dim)
                ).to(flattened.device)
                # Initialize with random weights and freeze them
                with torch.no_grad():
                    for param in self._dynamic_target.parameters():
                        param.data.normal_(0.0, 0.1)
                        param.requires_grad = False  # Freeze the weights
            return self._dynamic_target(flattened)
        
        # Use original target network if sizes match
        return self.target_network(flattened)
