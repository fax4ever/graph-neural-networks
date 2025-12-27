import gzip
import json
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
from torch.utils.data import random_split
from pearl_gnn.hyper_param import HyperParam


def load_datasets(hp: HyperParam):
    # Load train and test datasets
    data_dir = Path(__file__).parent.parent.parent.parent / "A"
    train_path = data_dir / "train.json.gz"
    test_path = data_dir / "test.json.gz"

    print(f"\nLoading datasets from: {data_dir}")
    train_dataset = GraphDataset(train_path)
    print(f"Train dataset: {len(train_dataset)} graphs")
    test_dataset = GraphDataset(test_path)
    print(f"Test dataset: {len(test_dataset)} graphs")

    # Test accessing a sample graph from each dataset
    if len(train_dataset) > 0:
        sample_train = train_dataset[0]
        print(f"\nSample train graph:")
        print(f"  - num_nodes: {sample_train.num_nodes}")
        print(f"  - edge_index shape: {sample_train.edge_index.shape}")
        print(f"  - edge_attr: {sample_train.edge_attr.shape if sample_train.edge_attr is not None else None}")
        print(f"  - y (label): {sample_train.y}")

    if len(test_dataset) > 0:
        sample_test = test_dataset[0]
        print(f"\nSample test graph:")
        print(f"  - num_nodes: {sample_test.num_nodes}")
        print(f"  - edge_index shape: {sample_test.edge_index.shape}")
        print(f"  - edge_attr: {sample_test.edge_attr.shape if sample_test.edge_attr is not None else None}")
        print(f"  - y (label): {sample_test.y}")

    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(hp.seed)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset, test_dataset


class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        self.num_graphs, self.graphs_dicts = self._count_graphs()
        super().__init__(None, transform, pre_transform)

    def len(self):
        return self.num_graphs

    def get(self, idx):
        return dictToGraphObject(self.graphs_dicts[idx])

    def _count_graphs(self):
        with gzip.open(self.raw, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)  # Load full JSON array without keeping references
            return len(graphs_dicts),graphs_dicts  # Return number of graphs


def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)