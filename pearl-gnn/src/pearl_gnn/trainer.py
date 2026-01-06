import logging
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import ZINC
from torch_geometric.utils import get_laplacian, to_dense_adj

from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.model.model_support import ModelSupport


# This code is adapted from:
# https://github.com/ehejin/Pearl-PE/blob/main/PEARL/src/zinc/trainer.py

# @fax4ever: 
# I tried to rewrite the code to be more readable and modular, simplify what is not necessary.
# In this process I could make some mistakes, so please check the code and report any issues.


class Trainer:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    hp: HyperParam
    model: ModelSupport


    def __init__(self, mf: ModelFactory):
        root = str(Path(__file__).parent.parent.parent / "data" / "ZINC")

        train_dataset: Dataset = ZINC(root=root, subset=mf.hp.use_subset, split="train", transform=self.get_lap)
        val_dataset: Dataset = ZINC(root=root, subset=mf.hp.use_subset, split="val", transform=self.get_lap)
        test_dataset: Dataset = ZINC(root=root, subset=mf.hp.use_subset, split="test", transform=self.get_lap)

        self.train_loader = DataLoader(train_dataset, batch_size=mf.hp.train_batch_size, shuffle=True, num_workers=3)
        self.val_loader = DataLoader(val_dataset, batch_size=mf.hp.val_batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=mf.hp.test_batch_size, shuffle=False, num_workers=0)
        self.hp = mf.hp
        self.model = ModelSupport(mf, self.lr_lambda)


    def train_all_epochs(self):
        logging.info(f"Total parameters: {sum(param.numel() for param in self.model.parameters())}")
        logging.info(f"Total training steps: {self.n_total_steps}")
        logging.info("Optimizer groups:\n" + "\n".join(group["name"] for group in self.model.optimizer.param_groups) + "\n")

        for self.curr_epoch in range(self.hp.num_epochs):
            train_loss, _ = self.model.train_epoch(self.train_loader)
            val_loss, _ = self.model.evaluate_epoch(self.val_loader)
            test_loss, _ = self.model.evaluate_epoch(self.test_loader)
            self.model.append_epoch_data(train_loss, val_loss, test_loss)

        self.model.plot_training()


    def lr_lambda(self, curr_step: int) -> float:
        """
        Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/optimization.py#L79
        """
        if curr_step < self.hp.n_warmup_steps:
            return curr_step / max(1, self.hp.n_warmup_steps)
        else:
            return max(0.0, (self.n_total_steps - curr_step) / max(1, self.n_total_steps - self.hp.n_warmup_steps))


    def get_lap(self, instance: Data) -> Data:
        n = instance.num_nodes
        L_edge_index, L_values = get_laplacian(instance.edge_index, normalization="sym", num_nodes=n)   # [2, X], [X]
        L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(dim=0)
        instance.Lap = L
        return instance


    @property
    def n_total_steps(self) -> int:
        return len(self.train_loader) * self.hp.num_epochs


