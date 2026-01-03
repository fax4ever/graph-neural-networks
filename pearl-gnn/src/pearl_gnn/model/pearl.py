from typing import List, Dict, Any
import torch
from torch import nn
from torch_geometric.data import Batch
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.model.pe import PEARLPositionalEncoder
from pearl_gnn.model.gine import GINEBaseModel

# This code is adapted from:
# https://github.com/ehejin/Pearl-PE/blob/main/PEARL/src/model.py

# @fax4ever: 
# I tried to rewrite the code to be more readable and modular, simplify what is not necessary.
# In this process I could make some mistakes, so please check the code and report any issues.


class PEARL_GNN_Model(nn.Module):
    node_features: nn.Embedding
    positional_encoding: PEARLPositionalEncoder
    pe_embedding: nn.Linear
    base_model: GINEBaseModel
    
    
    def __init__(self, mf: ModelFactory):
        super(PEARL_GNN_Model, self).__init__()
        self.node_features = nn.Embedding(mf.hp.n_node_types, mf.hp.node_emb_dims)
        self.positional_encoding = PEARLPositionalEncoder(mf)
        self.pe_embedding = nn.Linear(self.positional_encoding.out_dims, mf.hp.node_emb_dims)
        self.base_model = GINEBaseModel(mf)
        

    def forward(self, batch: Batch, W) -> torch.Tensor:
        X_n = self.node_features(batch.x.squeeze(dim=1))
        PE = self.positional_encoding(batch)
        X_n = X_n + self.pe_embedding(PE)
        return self.base_model(X_n, batch.edge_index, batch.edge_attr, PE, batch.batch)


    def get_param_groups(self) -> List[Dict[str, Any]]:
        return [{"name": name, "params": [param]} for name, param in self.named_parameters()]    