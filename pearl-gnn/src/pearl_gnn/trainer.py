import logging
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
from omegaconf import OmegaConf

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


    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, mf: ModelFactory):
        self.train_loader = DataLoader(train_dataset, batch_size=mf.hp.train_batch_size, shuffle=True, num_workers=3)
        self.val_loader = DataLoader(val_dataset, batch_size=mf.hp.val_batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=mf.hp.test_batch_size, shuffle=False, num_workers=0)
        self.hp = mf.hp
        self.model = ModelSupport(mf, self.lr_lambda)


    def train(self):
        logging.info("Configuration:\n" + OmegaConf.to_yaml(self.hp))
        logging.info(f"Total parameters: {sum(param.numel() for param in self.model.parameters())}")
        logging.info(f"Total training steps: {self.n_total_steps}")
        logging.info("Optimizer groups:\n" + "\n".join(group["name"] for group in self.model.optimizer.param_groups) + "\n")


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


