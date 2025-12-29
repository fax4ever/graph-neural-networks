from typing import Optional
from torch_geometric.nn import MessagePassing
import torch
from torch import nn
from torch.nn import functional as F
from pearl_gnn.model.mlp import MLP
from pearl_gnn.model.model_factory import ModelFactory
from torch_geometric.nn import global_mean_pool, global_add_pool
from typing import Callable


# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/gine.py 
# https://github.com/ehejin/Pearl-PE/blob/main/PEARL/src/gine.py

# @fax4ever: 
# I tried to rewrite the code to be more readable and modular, simplify what is not necessary.
# In this process I could make some mistakes, so please check the code and report any issues.


class GINELayer(MessagePassing):
    edge_features: nn.Embedding
    pe_embedding: MLP
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, mf: ModelFactory, in_dims: int, out_dims: int):
        super(GINELayer, self).__init__(aggr = "add") # for GIN aggregation function is always add
        self.edge_features = nn.Embedding(mf.hp.n_edge_types+1, mf.hp.in_dims)
        self.pe_embedding = mf.create_mlp(mf.hp.pe_emb, in_dims)
        self.eps = nn.Parameter(data=torch.randn(1))
        self.mlp = mf.create_mlp(in_dims, out_dims)
        
    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                PE: torch.Tensor) -> torch.Tensor:
        X_e = self.edge_features(edge_attr)
        X_e = X_e * self.pe_embedding(PE)
        S = self.propagate(edge_index, X=X_n, X_e=X_e)
        Z = (1 + self.eps) * X_n
        Z = Z + S
        return self.mlp(Z)

    def message(self, X_j: torch.Tensor, X_e: torch.Tensor) -> torch.Tensor:
        return F.relu(X_j + X_e)


class GINE(nn.Module):
    layers: nn.ModuleList
    batch_norms: Optional[nn.ModuleList]

    def __init__(self, mf: ModelFactory):
        super(GINE, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if mf.hp.gine_model_bn else None

        in_dims = mf.hp.node_emb_dims
        for _ in range(mf.hp.n_base_layers - 1):
            self.layers.append(GINELayer(mf, in_dims, mf.hp.base_hidden_dims))
            in_dims = mf.hp.base_hidden_dims
            if self.batch_norms is not None:
                self.batch_norms.append(nn.BatchNorm1d(mf.hp.base_hidden_dims))

        self.layers.append(GINELayer(mf, mf.hp.base_hidden_dims, mf.hp.base_hidden_dims))

    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            X_0 = X_n
            X_n = layer(X_n, edge_index, edge_attr, PE)

            # batch normalization
            if self.batch_norms is not None and i < len(self.layers) - 1:
                X_n = self.batch_norms[i](X_n)
            # residual connection        
            X_n = X_n + X_0
        return X_n               


class GINEBaseModel(nn.Module):
    gine: GINE
    mlp: MLP
    pooling: Callable

    def __init__(self, mf: ModelFactory):
        super(GINEBaseModel, self).__init__()
        self.gine = GINE(mf)
        self.mlp = mf.create_mlp(mf.hp.base_hidden_dims, mf.hp.out_dims)
        self.pooling = global_mean_pool if mf.hp.pooling == 'mean' else global_add_pool

    def forward(
        self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, PE: torch.Tensor, snorm: torch.Tensor,
            batch: torch.Tensor
    ) -> torch.Tensor:
        X_n = self.gine(X_n, edge_index, edge_attr, PE)  
        X_n = self.pooling(X_n, batch) 
        Y_pred = self.mlp(X_n)        
        return Y_pred.squeeze(dim=1)    