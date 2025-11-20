import torch
import torch.nn as nn


class BCNetwork(nn.Module):
    """
    Behavior Cloning Network
    Input: 72 dimensions
    Output: 12 dimensions
    """
    def __init__(self, input_dim=72, output_dim=12, hidden_dims=[256, 256, 128]):
        """
        Args:
            input_dim: Input dimension (default: 72)
            output_dim: Output dimension (default: 12)
            hidden_dims: List of hidden layer dimensions (default: [256, 256, 128])
        """
        super(BCNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

