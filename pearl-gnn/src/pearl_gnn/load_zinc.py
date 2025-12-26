from torch_geometric.datasets import ZINC
from torch_geometric.data import Dataset
from pathlib import Path
from typing import Tuple

from pearl_gnn.hyper_param import HyperParam


def load_datasets(hp: HyperParam, root: str | None = None) -> Tuple[Dataset, Dataset]:
    """
    Load the ZINC dataset from torch_geometric.

    Args:
        hp: HyperParam instance containing configuration (e.g., use_subset).
        root: Root directory where the dataset should be saved.
              Defaults to a 'data/ZINC' folder in the project root.

    Returns:
        Tuple of (train_dataset, test_dataset) of type torch_geometric.data.Dataset
    """
    if root is None:
        root = str(Path(__file__).parent.parent.parent / "data" / "ZINC")

    print(f"\nLoading ZINC dataset from: {root}")
    print(f"Using {'subset (12K)' if hp.use_subset else 'full dataset (250K)'}")

    train_dataset: Dataset = ZINC(root=root, subset=hp.use_subset, split="train")
    val_dataset: Dataset = ZINC(root=root, subset=hp.use_subset, split="val")
    test_dataset: Dataset = ZINC(root=root, subset=hp.use_subset, split="test")

    print(f"Train dataset: {len(train_dataset)} graphs")
    print(f"Test dataset: {len(test_dataset)} graphs")

    # Display sample graph info
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\nSample graph structure:")
        print(f"  - num_nodes: {sample.num_nodes}")
        print(f"  - x (node features) shape: {sample.x.shape if sample.x is not None else None}")
        print(f"  - edge_index shape: {sample.edge_index.shape}")
        print(f"  - edge_attr shape: {sample.edge_attr.shape if sample.edge_attr is not None else None}")
        print(f"  - y (target): {sample.y}")

    return train_dataset, val_dataset, test_dataset

