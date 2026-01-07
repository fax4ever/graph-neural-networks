from math import sin
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.datasets import ZINC
from torch_geometric.utils import get_laplacian, to_dense_adj
from pathlib import Path
from pearl_gnn.model.pearl import PEARL_GNN_Model
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.model.model_support import ModelSupport


ROOT = str(Path(__file__).parent.parent / "data" / "ZINC")
SUB_DATASET_SIZE = 10
hyper_param = HyperParam()
N_TOTAL_STEPS = SUB_DATASET_SIZE * hyper_param.num_epochs


def get_lap(instance: Data) -> Data:
    n = instance.num_nodes
    L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)   # [2, X], [X]
    L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(dim=0)
    instance.Lap = L
    return instance


def sparse_lap(instance: Data) -> Data:
    n = instance.num_nodes
    L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)
    instance.L_edge_index = L_edge_index
    instance.L_values = L_values
    instance.max_num_nodes = n
    return instance


def dense_lap(L_edge_index, L_values, max_num_nodes):
    return to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=max_num_nodes).squeeze(dim=0)


def lr_lambda(curr_step: int) -> float:
    """
    Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/optimization.py#L79
    """
    if curr_step < hyper_param.n_warmup_steps:
        return curr_step / max(1, hyper_param.n_warmup_steps)
    else:
        return max(0.0, (N_TOTAL_STEPS - curr_step) / max(1, N_TOTAL_STEPS - hyper_param.n_warmup_steps))


class TestModel:
    def test_model_creation(self):
        model = PEARL_GNN_Model(ModelFactory(HyperParam()))
        assert model is not None


    def test_dataset_enumeration(self):
        train_dataset: Dataset = ZINC(root=ROOT, subset=True, split="train")
        size = len(train_dataset)
        for i in range(size):
            item = train_dataset[i]
            get_lap(item)
            assert item.Lap is not None


    def test_dataset_transformation(self):
        train_dataset: Dataset = ZINC(root=ROOT, subset=True, split="train", transform=get_lap)
        size = len(train_dataset)
        for i in range(size):
            item = train_dataset[i]
            assert item.Lap is not None


    def test_dataloader_transformation(self):
        train_dataset: Dataset = ZINC(root=ROOT, subset=True, split="train", transform=sparse_lap)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        for batch in train_loader:
            assert batch.L_edge_index is not None
            assert batch.L_values is not None
            assert batch.max_num_nodes is not None
            # L = dense_lap(batch.L_edge_index, batch.L_values, batch.max_num_nodes)
            # assert L is not None


    def test_subdataset_training(self):
        dataset: Dataset = ZINC(root=ROOT, subset=True, split="train", transform=sparse_lap)
        dataset = dataset[:SUB_DATASET_SIZE]
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model_factory = ModelFactory(hyper_param)
        model = ModelSupport(model_factory, lr_lambda)

        # model.train_epoch(loader)

