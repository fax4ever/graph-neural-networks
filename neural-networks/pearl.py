"""
PEARL: Positional Encodings Augmented with Representation Learning
Implementation based on "Learning Efficient Positional Encodings with Graph Neural Networks"
arXiv:2502.01122

This module implements learnable positional encodings for graphs using message-passing GNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Callable


class PEARLConv(MessagePassing):
    """
    PEARL Convolutional Layer
    
    Implements a message-passing layer that generates positional encodings
    by operating on initialized node features.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
        **kwargs
    ):
        super(PEARLConv, self).__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear transformation for message passing
        self.lin_msg = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_msg.reset_parameters()
        self.lin_self.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of PEARL convolution
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
        """
        # Add self-loops
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=x.size(0)
        )
        
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        if edge_weight is not None:
            norm = norm * edge_weight
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.lin_self(x)
        
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def message(self, x_j, norm):
        """Message computation"""
        return norm.view(-1, 1) * self.lin_msg(x_j)


class StatisticalPooling(nn.Module):
    """
    Statistical Pooling for Permutation Equivariance
    
    Computes multiple statistical moments to maintain permutation equivariance
    while providing expressive representations.
    """
    
    def __init__(self, mode: str = 'all'):
        """
        Args:
            mode: Type of pooling ('mean', 'max', 'std', 'all')
        """
        super(StatisticalPooling, self).__init__()
        self.mode = mode
        
    def forward(self, x):
        """
        Args:
            x: Node features [num_nodes, num_features]
            
        Returns:
            Pooled features maintaining equivariance
        """
        if self.mode == 'mean':
            return x.mean(dim=0, keepdim=True).expand_as(x)
        elif self.mode == 'max':
            return x.max(dim=0, keepdim=True)[0].expand_as(x)
        elif self.mode == 'std':
            return x.std(dim=0, keepdim=True).expand_as(x)
        elif self.mode == 'all':
            mean_pool = x.mean(dim=0, keepdim=True)
            max_pool = x.max(dim=0, keepdim=True)[0]
            std_pool = x.std(dim=0, keepdim=True)
            pooled = torch.cat([mean_pool, max_pool, std_pool], dim=-1)
            return pooled.expand(x.size(0), -1)
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")


class PEARL(nn.Module):
    """
    PEARL: Positional Encodings Augmented with Representation Learning
    
    Main module that generates learnable positional encodings for graphs
    using message-passing GNNs with linear complexity.
    
    Key properties:
    - Stability: Robust to graph perturbations
    - Expressive Power: Can approximate equivariant functions of eigenvectors
    - Scalability: Linear complexity O(n)
    - Genericness: Works across different graph types
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        pe_dim: int = 32,
        num_layers: int = 3,
        init_mode: str = 'random',
        pooling_mode: str = 'all',
        activation: Callable = F.relu,
        dropout: float = 0.0,
        use_batch_norm: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden dimension for GNN layers
            pe_dim: Output positional encoding dimension
            num_layers: Number of message-passing layers
            init_mode: Node initialization ('random' or 'basis')
            pooling_mode: Statistical pooling type
            activation: Activation function
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(PEARL, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pe_dim = pe_dim
        self.num_layers = num_layers
        self.init_mode = init_mode
        self.activation = activation
        self.dropout = dropout
        
        # Statistical pooling for equivariance
        self.pooling = StatisticalPooling(mode=pooling_mode)
        
        # Determine input dimension based on pooling mode
        if pooling_mode == 'all':
            pool_multiplier = 3  # mean, max, std
        else:
            pool_multiplier = 1
            
        # Input projection
        self.input_proj = nn.Linear(hidden_dim * pool_multiplier, hidden_dim)
        
        # Message-passing layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(num_layers):
            self.convs.append(PEARLConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection to PE dimension
        self.output_proj = nn.Linear(hidden_dim, pe_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters"""
        self.input_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norms is not None:
            for bn in self.batch_norms:
                bn.reset_parameters()
        self.output_proj.reset_parameters()
    
    def initialize_features(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Initialize node features for PE generation
        
        Args:
            num_nodes: Number of nodes in the graph
            device: Device to create tensor on
            
        Returns:
            Initialized node features [num_nodes, hidden_dim]
        """
        if self.init_mode == 'random':
            # Random Gaussian initialization
            features = torch.randn(num_nodes, self.hidden_dim, device=device)
            # Normalize for stability
            features = F.normalize(features, p=2, dim=1)
        elif self.init_mode == 'basis':
            # Standard basis vectors (cyclic)
            features = torch.zeros(num_nodes, self.hidden_dim, device=device)
            for i in range(num_nodes):
                features[i, i % self.hidden_dim] = 1.0
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
            
        return features
    
    def forward(self, edge_index, num_nodes: Optional[int] = None, edge_weight=None):
        """
        Generate positional encodings for a graph
        
        Args:
            edge_index: Graph connectivity [2, num_edges]
            num_nodes: Number of nodes (inferred if None)
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Positional encodings [num_nodes, pe_dim]
        """
        device = edge_index.device
        
        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1
        
        # Initialize node features
        x = self.initialize_features(num_nodes, device)
        
        # Apply statistical pooling for equivariance
        x = self.pooling(x)
        
        # Input projection
        x = self.input_proj(x)
        x = self.activation(x)
        
        # Message-passing layers
        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index, edge_weight)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_prev
        
        # Project to PE dimension
        pe = self.output_proj(x)
        
        # Normalize PEs for stability
        pe = F.normalize(pe, p=2, dim=1)
        
        return pe


class PEARLWithLaplacianInit(PEARL):
    """
    PEARL variant with Laplacian-based initialization
    
    Uses graph structure information (degree) for initialization
    to provide better structural awareness.
    """
    
    def initialize_features(self, num_nodes: int, device: torch.device, 
                          edge_index=None) -> torch.Tensor:
        """
        Initialize features using graph Laplacian properties
        
        Args:
            num_nodes: Number of nodes
            device: Device to create tensor on
            edge_index: Graph connectivity for degree computation
        """
        features = torch.randn(num_nodes, self.hidden_dim, device=device)
        
        if edge_index is not None:
            # Incorporate degree information
            row, col = edge_index
            deg = degree(col, num_nodes, dtype=features.dtype)
            deg_normalized = deg / (deg.max() + 1e-6)
            
            # Scale features by degree
            features = features * deg_normalized.unsqueeze(1)
        
        # Normalize for stability
        features = F.normalize(features, p=2, dim=1)
        return features
    
    def forward(self, edge_index, num_nodes: Optional[int] = None, edge_weight=None):
        """Forward pass with Laplacian initialization"""
        device = edge_index.device
        
        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1
        
        # Initialize with degree information
        x = self.initialize_features(num_nodes, device, edge_index)
        
        # Rest is same as base PEARL
        x = self.pooling(x)
        x = self.input_proj(x)
        x = self.activation(x)
        
        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index, edge_weight)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            if i > 0:
                x = x + x_prev
        
        pe = self.output_proj(x)
        pe = F.normalize(pe, p=2, dim=1)
        
        return pe

