
from typing import List
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.model.gin import GIN
from pearl_gnn.model.mlp import MLP

# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/stable_expressive_pe.py 
# https://github.com/ehejin/Pearl-PE/blob/main/PEARL/src/pe.py

# @fax4ever: 
# I tried to rewrite the code to be more readable and modular, simplify what is not necessary.
# In this process I could make some mistakes, so please check the code and report any issues.


def initial_pe(batch: Batch, basis: str, num_samples: int):
    W_list = []
    for i in range(len(batch.Lap)):
        if basis:
            W = torch.eye(batch.Lap[i].shape[0])
        else:
            W = torch.randn(batch.Lap[i].shape[0], num_samples) #BxNxM
        W_list.append(W)



class RandomSampleAggregator(nn.Module):
    gin: GIN
    mlp: MLP
    running_sum: torch.Tensor

    def __init__(self, mf: ModelFactory):
        super(BasisSampleAggregator, self).__init__()
        self.gin = GIN(mf)
        self.mlp = mf.create_mlp(mf.hp.pe_dims, mf.hp.pe_dims)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor, final=True) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        W = torch.cat(W_list, dim=0)  
        PE = self.gin(W, edge_index)  
        PE = PE.sum(dim=1)
        self.running_sum += PE
        if final:
            PE = self.running_sum
            self.running_sum = 0
        return PE

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims


class BasisSampleAggregator(nn.Module):
    gin: GIN
    mlp: MLP

    def __init__(self, mf: ModelFactory):
        super(BasisSampleAggregator, self).__init__()
        self.gin = GIN(mf)
        self.mlp = mf.create_mlp(mf.hp.pe_dims, mf.hp.pe_dims)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []     # [N_i, N_max, M] * B
        mask = [] # node masking, [N_i, N_max] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)

            # >>> (torch.arange(7) < 3).tile((3, 1))
            # tensor([[ True,  True,  True, False, False, False, False],
            #         [ True,  True,  True, False, False, False, False],
            #         [ True,  True,  True, False, False, False, False]])
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]
        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]

        # >>> torch.cat([(torch.arange(7) < 3).tile((3, 1)), (torch.arange(7) < 4).tile((4, 1))], dim=0)
        # tensor([[ True,  True,  True, False, False, False, False],
        #         [ True,  True,  True, False, False, False, False],
        #         [ True,  True,  True, False, False, False, False],
        #         [ True,  True,  True,  True, False, False, False],
        #         [ True,  True,  True,  True, False, False, False],
        #         [ True,  True,  True,  True, False, False, False],
        #         [ True,  True,  True,  True, False, False, False]])
        mask = torch.cat(mask, dim=0)   # [N_sum, N_max]

        PE = self.gin(W, edge_index, mask=mask)       # [N_sum, N_max, D_pe]
        PE = (PE * mask.unsqueeze(-1)).sum(dim=1)
        return PE

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims


def filter(S, W, k):
    # S is laplacian and W is NxN e or NxM x_m
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(k-1): 
        out = S @ out # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1) #NxMxK


class PEARLPositionalEncoder:
    sample_aggr: nn.Module
    layers: nn.ModuleList
    norms: nn.ModuleList
    activation: nn.ReLU

    def __init__(self, mf: ModelFactory):
        super(PEARLPositionalEncoder, self).__init__()
        self.sample_aggr = BasisSampleAggregator(mf) if mf.hp.basis else RandomSampleAggregator(mf)
        self.layers = nn.ModuleList([nn.Linear(mf.hp.pearl_k if i==0 else mf.hp.pearl_k, 
            mf.hp.pearl_mlp_hid if i<mf.hp.pearl_mlp_nlayers-1 else mf.hp.pearl_mlp_out) 
            for i in range(mf.hp.pearl_mlp_nlayers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(mf.hp.pearl_mlp_hid if i<mf.hp.pearl_mlp_nlayers-1 else mf.hp.pearl_mlp_out) 
            for i in range(mf.hp.pearl_mlp_nlayers)])
        self.activation = nn.ReLU()
        self.k = mf.hp.pearl_k
        self.basis = mf.hp.basis
        self.num_samples = mf.hp.num_samples

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        :param batch: current batch to process
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        Lap = batch.Lap # Laplacian
        edge_index = batch.edge_index # Graph connectivity in COO format. [2, E_sum]
        W = initial_pe(batch, self.basis, self.num_samples) #  B*[NxM] or BxNxN

        W_list = []
        for lap, w in zip(Lap, W):
            output = filter(lap, w, self.k)   # output [NxMxK]
            if self.mlp_nlayers > 0:
                for layer, bn in zip(self.layers, self.norms):
                    output = output.transpose(0, 1)
                    output = layer(output)
                    output = bn(output.transpose(1,2)).transpose(1,2)
                    output = self.activation(output)
                    output = output.transpose(0, 1)
            W_list.append(output)             # [NxMxK]*B
        return self.sample_aggr(W_list, edge_index, self.basis)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.sample_aggr.out_dims
            
