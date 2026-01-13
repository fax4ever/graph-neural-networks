from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.datasets import ZINC

from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.model.model_support import ModelSupport
from pearl_gnn.model.pe import add_laplacian_transform


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

        # Apply the laplacian transform to pre-compute sparse Laplacian components
        # that can be properly batched by PyG
        train_dataset: Dataset = ZINC(root=root, subset=mf.hp.use_subset, split="train", transform=add_laplacian_transform)
        val_dataset: Dataset = ZINC(root=root, subset=mf.hp.use_subset, split="val", transform=add_laplacian_transform)
        test_dataset: Dataset = ZINC(root=root, subset=mf.hp.use_subset, split="test", transform=add_laplacian_transform)

        self.train_loader = DataLoader(train_dataset, batch_size=mf.hp.train_batch_size, shuffle=True, num_workers=3)
        self.val_loader = DataLoader(val_dataset, batch_size=mf.hp.val_batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=mf.hp.test_batch_size, shuffle=False, num_workers=0)
        self.hp = mf.hp
        self.model = ModelSupport(mf, self.lr_lambda)


    def train_all_epochs(self):
        print(f"Total parameters: {sum(param.numel() for param in self.model.parameters()):,}")
        print(f"Total training steps: {self.n_total_steps:,}")
        print(f"Training for {self.hp.num_epochs} epochs...")
        print("-" * 60)

        for self.curr_epoch in range(self.hp.num_epochs):
            train_loss, _ = self.model.train_epoch(self.train_loader, self.curr_epoch, self.hp.num_epochs)
            val_loss, _ = self.model.evaluate_epoch(self.val_loader, desc="Val")
            test_loss, _ = self.model.evaluate_epoch(self.test_loader, desc="Test")
            self.model.append_epoch_data(train_loss, val_loss, test_loss, self.curr_epoch, self.hp.num_epochs)

        self.model.plot_training()


    def lr_lambda(self, curr_step: int) -> float:
        """
        Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/optimization.py#L79
        """
        if curr_step < self.hp.n_warmup_steps:
            return curr_step / max(1, self.hp.n_warmup_steps)
        else:
            return max(0.0, (self.n_total_steps - curr_step) / max(1, self.n_total_steps - self.hp.n_warmup_steps))


    @property
    def n_total_steps(self) -> int:
        return len(self.train_loader) * self.hp.num_epochs


