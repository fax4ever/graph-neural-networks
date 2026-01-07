from math import sin
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.datasets import ZINC
from pathlib import Path
from pearl_gnn.model.pearl import PEARL_GNN_Model
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.hyper_param import HyperParam
from torch_geometric.utils import get_laplacian, to_dense_adj


ROOT = str(Path(__file__).parent.parent / "data" / "ZINC")


def get_lap(instance: Data) -> Data:
    n = instance.num_nodes
    L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)   # [2, X], [X]
    L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(dim=0)
    instance.Lap = L
    return instance


def simple_lap(instance: Data) -> Data:
    n = instance.num_nodes
    L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)
    instance.L_edge_index = L_edge_index
    instance.L_values = L_values
    return instance


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
        train_dataset: Dataset = ZINC(root=ROOT, subset=True, split="train", transform=simple_lap)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        for batch in train_loader:
            assert batch.L_edge_index is not None
            assert batch.L_values is not None
