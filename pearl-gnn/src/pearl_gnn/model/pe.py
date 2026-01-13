
from typing import List
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import get_laplacian, to_dense_adj
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.model.gin import GIN
from pearl_gnn.model.mlp import MLP

# This code is adapted from:
# https://github.com/Graph-COM/SPE/blob/master/src/stable_expressive_pe.py 
# https://github.com/ehejin/Pearl-PE/blob/main/PEARL/src/pe.py

# @fax4ever: 
# I tried to rewrite the code to be more readable and modular, simplify what is not necessary.
# In this process I could make some mistakes, so please check the code and report any issues.


def add_laplacian_transform(data: Data) -> Data:
    """
    Transform to pre-compute sparse Laplacian components for each graph.
    Use this as a transform when loading the dataset:
        dataset = ZINC(root=root, transform=add_laplacian_transform)
    
    The sparse representation can be properly collated by PyG's Batch.
    """
    n = data.num_nodes
    L_edge_index, L_values = get_laplacian(data.edge_index, normalization="sym", num_nodes=n)
    data.lap_edge_index = L_edge_index
    data.lap_edge_attr = L_values
    return data


def get_per_graph_dense_laplacians(batch: Batch) -> List[torch.Tensor]:
    """
    Reconstruct per-graph dense Laplacians from a batched graph.
    
    After batching, PyG merges graphs into one large graph. This function
    uses the batch's slicing information to extract each graph's Laplacian
    and convert it to a dense matrix.
    
    Args:
        batch: A PyG Batch object with lap_edge_index and lap_edge_attr 
               (added via add_laplacian_transform)
    
    Returns:
        List of dense Laplacian matrices, one per graph in the batch.
    """
    # Get the number of graphs in the batch
    num_graphs = batch.num_graphs
    
    # batch.ptr gives us cumulative node counts: [0, n1, n1+n2, ...]
    # This tells us where each graph's nodes start/end
    ptr = batch.ptr
    
    # Get slicing info for lap_edge_index (edge offsets and increments)
    # PyG stores this automatically when batching
    lap_edge_slices = batch._slice_dict['lap_edge_index']  # edge index boundaries
    lap_attr_slices = batch._slice_dict['lap_edge_attr']   # edge attr boundaries
    
    laplacians = []
    for i in range(num_graphs):
        # Number of nodes in this graph
        n_nodes = ptr[i + 1] - ptr[i]
        node_offset = ptr[i].item()
        
        # Extract this graph's Laplacian edges
        edge_start = lap_edge_slices[i].item()
        edge_end = lap_edge_slices[i + 1].item()
        
        # Get the edge indices for this graph and remove the node offset
        graph_lap_edge_index = batch.lap_edge_index[:, edge_start:edge_end] - node_offset
        
        # Get the edge values for this graph
        attr_start = lap_attr_slices[i].item()
        attr_end = lap_attr_slices[i + 1].item()
        graph_lap_edge_attr = batch.lap_edge_attr[attr_start:attr_end]
        
        # Convert to dense
        dense_lap = to_dense_adj(
            graph_lap_edge_index, 
            edge_attr=graph_lap_edge_attr, 
            max_num_nodes=n_nodes
        ).squeeze(0)
        
        laplacians.append(dense_lap)
    
    return laplacians


def initial_pe(batch: Batch, basis: bool, num_samples: int) -> List[torch.Tensor]:
    """
    Initialize positional encoding basis for each graph in the batch.
    
    Args:
        batch: PyG Batch object
        basis: If True, use identity matrix (deterministic). 
               If False, use random samples (stochastic).
        num_samples: Number of random samples (only used if basis=False)
    
    Returns:
        List of initial PE matrices, one per graph [N_i x N_i] or [N_i x M]
    """
    ptr = batch.ptr
    num_graphs = batch.num_graphs
    device = batch.x.device
    
    W_list = []
    for i in range(num_graphs):
        n_nodes = (ptr[i + 1] - ptr[i]).item()
        if basis:
            W = torch.eye(n_nodes, device=device)  # [N_i x N_i]
        else:
            W = torch.randn(n_nodes, num_samples, device=device)  # [N_i x M]
        W_list.append(W)
    
    return W_list


class RandomSampleAggregator(nn.Module):
    gin: GIN
    mlp: MLP
    running_sum: torch.Tensor = 0

    def __init__(self, mf: ModelFactory):
        super(RandomSampleAggregator, self).__init__()
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


class PEARLPositionalEncoder(nn.Module):
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
        :param batch: current batch to process (must have lap_edge_index and lap_edge_attr
                      from add_laplacian_transform)
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        # Get per-graph dense Laplacians from the batched sparse representation
        Lap_list = get_per_graph_dense_laplacians(batch)
        edge_index = batch.edge_index  # Graph connectivity in COO format. [2, E_sum]
        W_init = initial_pe(batch, self.basis, self.num_samples)  # B*[NxM] or B*[NxN]

        W_list = []
        for lap, w in zip(Lap_list, W_init):
            output = filter(lap, w, self.k)   # output [NxMxK]
            if len(self.layers) > 0:
                for layer, bn in zip(self.layers, self.norms):
                    output = output.transpose(0, 1)
                    output = layer(output)
                    output = bn(output.transpose(1, 2)).transpose(1, 2)
                    output = self.activation(output)
                    output = output.transpose(0, 1)
            W_list.append(output)             # [NxMxK]*B
        return self.sample_aggr(W_list, edge_index)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.sample_aggr.out_dims
            
