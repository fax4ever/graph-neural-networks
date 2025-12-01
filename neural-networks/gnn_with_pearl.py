"""
Graph Neural Network Models with PEARL Positional Encodings

Implements various GNN architectures that leverage PEARL for enhanced
positional awareness and expressiveness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool
from typing import Optional, Callable

from pearl import PEARL, PEARLWithLaplacianInit


class GNNWithPEARL(nn.Module):
    """
    Base GNN model augmented with PEARL positional encodings
    
    This model combines structural graph learning with learned positional encodings
    to achieve better representation power for graph-level and node-level tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pe_dim: int = 32,
        num_layers: int = 3,
        gnn_type: str = 'gcn',
        pe_init_mode: str = 'random',
        pe_num_layers: int = 2,
        dropout: float = 0.5,
        pooling: str = 'mean',
        task: str = 'graph',  # 'graph' or 'node'
        use_pearl: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GNN
            output_dim: Output dimension (num classes)
            pe_dim: Positional encoding dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gcn', 'gat', 'gin')
            pe_init_mode: PEARL initialization mode
            pe_num_layers: Number of PEARL layers
            dropout: Dropout rate
            pooling: Graph pooling method ('mean', 'add')
            task: Task type ('graph' or 'node')
            use_pearl: Whether to use PEARL (for ablation studies)
        """
        super(GNNWithPEARL, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.pooling = pooling
        self.task = task
        self.use_pearl = use_pearl
        
        # PEARL positional encoding module
        if use_pearl:
            self.pearl = PEARL(
                hidden_dim=hidden_dim,
                pe_dim=pe_dim,
                num_layers=pe_num_layers,
                init_mode=pe_init_mode,
                dropout=dropout
            )
            # Combine input features with positional encodings
            effective_input_dim = input_dim + pe_dim
        else:
            self.pearl = None
            effective_input_dim = input_dim
        
        # Input projection
        self.input_proj = nn.Linear(effective_input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch=None, edge_weight=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector for graph-level tasks [num_nodes]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Output predictions
        """
        num_nodes = x.size(0)
        
        # Generate PEARL positional encodings
        if self.use_pearl:
            pe = self.pearl(edge_index, num_nodes, edge_weight)
            # Concatenate with input features
            x = torch.cat([x, pe], dim=-1)
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GNN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_prev = x
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_prev
        
        # Task-specific output
        if self.task == 'graph':
            # Graph-level pooling
            if batch is None:
                # Single graph
                if self.pooling == 'mean':
                    x = x.mean(dim=0, keepdim=True)
                elif self.pooling == 'add':
                    x = x.sum(dim=0, keepdim=True)
            else:
                # Batched graphs
                if self.pooling == 'mean':
                    x = global_mean_pool(x, batch)
                elif self.pooling == 'add':
                    x = global_add_pool(x, batch)
        
        # Output projection
        out = self.output_proj(x)
        
        return out


class TransformerWithPEARL(nn.Module):
    """
    Graph Transformer with PEARL Positional Encodings
    
    Combines self-attention mechanism with PEARL positional encodings
    for powerful graph representation learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pe_dim: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        pe_num_layers: int = 2,
        dropout: float = 0.1,
        task: str = 'graph'
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            output_dim: Output dimension
            pe_dim: Positional encoding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            pe_num_layers: Number of PEARL layers
            dropout: Dropout rate
            task: Task type ('graph' or 'node')
        """
        super(TransformerWithPEARL, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.task = task
        
        # PEARL positional encoding
        self.pearl = PEARL(
            hidden_dim=hidden_dim,
            pe_dim=pe_dim,
            num_layers=pe_num_layers,
            dropout=dropout
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim + pe_dim, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes]
            
        Returns:
            Output predictions
        """
        num_nodes = x.size(0)
        
        # Generate PEARL positional encodings
        pe = self.pearl(edge_index, num_nodes)
        
        # Concatenate features with positional encodings
        x = torch.cat([x, pe], dim=-1)
        x = self.input_proj(x)
        
        # For single graph, add batch dimension
        if batch is None:
            x = x.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            x = self.transformer(x)
            x = x.squeeze(0)  # [num_nodes, hidden_dim]
        else:
            # Handle batched graphs
            # Group nodes by graph
            batch_size = batch.max().item() + 1
            max_nodes = (batch.bincount()).max().item()
            
            # Pad to max nodes per graph
            x_batched = torch.zeros(batch_size, max_nodes, self.hidden_dim, 
                                   device=x.device, dtype=x.dtype)
            mask = torch.ones(batch_size, max_nodes, device=x.device, dtype=torch.bool)
            
            for i in range(batch_size):
                nodes_in_graph = (batch == i).nonzero(as_tuple=True)[0]
                num_nodes_in_graph = nodes_in_graph.size(0)
                x_batched[i, :num_nodes_in_graph] = x[nodes_in_graph]
                mask[i, :num_nodes_in_graph] = False
            
            # Apply transformer with padding mask
            x_batched = self.transformer(x_batched, src_key_padding_mask=mask)
            
            # Unpack batched representation
            x_list = []
            for i in range(batch_size):
                num_nodes_in_graph = (batch == i).sum().item()
                x_list.append(x_batched[i, :num_nodes_in_graph])
            x = torch.cat(x_list, dim=0)
        
        # Task-specific output
        if self.task == 'graph':
            if batch is None:
                x = x.mean(dim=0, keepdim=True)
            else:
                x = global_mean_pool(x, batch)
        
        out = self.output_proj(x)
        return out


class EnsemblePEARL(nn.Module):
    """
    Ensemble of PEARL models with different initialization strategies
    
    Combines multiple PEARL variants to improve robustness and performance.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pe_dim: int = 32,
        num_models: int = 3,
        **kwargs
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            pe_dim: Positional encoding dimension
            num_models: Number of ensemble models
            **kwargs: Additional arguments for GNNWithPEARL
        """
        super(EnsemblePEARL, self).__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList()
        
        # Create ensemble with different initialization strategies
        init_modes = ['random', 'basis']
        
        for i in range(num_models):
            init_mode = init_modes[i % len(init_modes)]
            model = GNNWithPEARL(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                pe_dim=pe_dim,
                pe_init_mode=init_mode,
                **kwargs
            )
            self.models.append(model)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass - averages predictions from all models
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch vector
            
        Returns:
            Averaged predictions
        """
        outputs = []
        for model in self.models:
            out = model(x, edge_index, batch)
            outputs.append(out)
        
        # Average predictions
        output = torch.stack(outputs, dim=0).mean(dim=0)
        return output

