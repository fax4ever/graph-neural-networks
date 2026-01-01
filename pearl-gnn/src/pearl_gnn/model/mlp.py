import torch
from torch import nn
from pearl_gnn.hyper_param import HyperParam
from typing import Optional

# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/mlp.py 
# https://github.com/ehejin/Pearl-PE/blob/main/PEARL/src/mlp.py
# https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP

# @fax4ever: 
# I tried to rewrite the code to be more readable and modular, simplify what is not necessary.
# In this process I could make some mistakes, so please check the code and report any issues.


class MLP(nn.Module):
    layers: nn.ModuleList
    linear: nn.Linear
    dropout: nn.Dropout

    def __init__(self, in_dims: int, out_dims: int, hp: HyperParam):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(hp.n_mlp_layers - 1):
            self.layers.append(MLPLayer(in_dims, hp.mlp_hidden_dims, hp))
            in_dims = hp.mlp_hidden_dims

        self.linear = nn.Linear(hp.mlp_hidden_dims, out_dims)
        self.dropout = nn.Dropout(p=hp.mlp_dropout_prob)

    def forward(self, X: torch.Tensor, mask=None) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, mask=mask)
        X = self.linear(X)
        X = self.dropout(X)
        return X

    @property
    def out_dims(self) -> int:
        return self.linear.out_features


class MLPLayer(nn.Module):
    linear: nn.Linear
    normlaization: Optional[nn.Module]
    activation: nn.ReLU
    dropout: nn.Dropout
    
    def __init__(self, in_dims: int, out_dims: int, hp: HyperParam):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_dims, out_dims)

        if hp.mlp_norm_type == "batch":
            self.normlaization = nn.BatchNorm1d(out_dims)
        else:
            self.normlaization = nn.LayerNorm(out_dims)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=hp.mlp_dropout_prob)

    def forward(self, X: torch.Tensor, mask=None) -> torch.Tensor:
        X = self.linear(X)
        if mask is not None:
            X[~mask] = 0

        if mask is None:
            shape = X.size()
            X = X.reshape(-1, shape[-1])
            X = self.normlaization(X)
            X = X.reshape(shape)
        else:
            X[mask] = self.normlaization(X[mask])    

        X = self.activation(X)
        X = self.dropout(X)
        return X

