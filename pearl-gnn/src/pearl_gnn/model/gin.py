from torch_geometric.nn import MessagePassing
import torch
from torch import nn
from torch.nn import functional as F
from pearl_gnn.model.mlp import MLP
from pearl_gnn.model.model_factory import ModelFactory
from typing import Optional


# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/gin.py 
# https://github.com/ehejin/Pearl-PE/blob/main/PEARL/src/gin.py

# @fax4ever: 
# I tried to rewrite the code to be more readable and modular, simplify what is not necessary.
# In this process I could make some mistakes, so please check the code and report any issues.


class GINLayer(MessagePassing):
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, mf: ModelFactory, in_dims: int, out_dims: int):
        super(GINLayer, self).__init__(aggr = "add") # for GIN aggregation function is always add
        self.eps = nn.Parameter(data=torch.randn(1))
        self.mlp = mf.create_mlp(in_dims, out_dims)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, mask=None) -> torch.Tensor:
        S = self.propagate(edge_index, X=X)
        Z = (1 + self.eps) * X
        Z = Z + S
        return self.mlp(Z, mask=mask)
    
    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        return F.relu(X_j)

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims    


class GIN(nn.Module):
    layers: nn.ModuleList
    batch_norms: Optional[nn.ModuleList]

    def __init__(self, mf: ModelFactory):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if mf.hp.gin_sample_aggregator_bn else None

        in_dims = mf.hp.pearl_mlp_out
        for _ in range(mf.hp.n_sample_aggr_layers - 1):
            self.layers.append(GINLayer(mf, in_dims, mf.hp.sample_aggr_hidden_dims))
            in_dims = mf.hp.sample_aggr_hidden_dims
            if self.batch_norms is not None:
                self.batch_norms.append(nn.BatchNorm1d(mf.hp.sample_aggr_hidden_dims))

        self.layers.append(GINLayer(mf, mf.hp.sample_aggr_hidden_dims, mf.hp.pe_dims))

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, mask=None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            X0 = X
            X = layer(X, edge_index, mask=mask)

            if mask is not None:
                X[~mask] = 0

            if self.batch_norms is not None and i < len(self.layers) - 1:
                assert not X.ndim == 3
                if mask is not None:
                    X[mask] = self.batch_norms[i](X[mask])
                else:
                    X = self.batch_norms[i](X)

            X = X + X0
        return X

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims

