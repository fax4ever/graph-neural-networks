import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
from omegaconf import OmegaConf
from pearl_gnn.helper.log_service import LogService
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.model.pearl import PEARL_GNN_Model

# This code is adapted from:
# https://github.com/ehejin/Pearl-PE/blob/main/PEARL/src/zinc/trainer.py

# @fax4ever: 
# I tried to rewrite the code to be more readable and modular, simplify what is not necessary.
# In this process I could make some mistakes, so please check the code and report any issues.


class Trainer:
    model: PEARL_GNN_Model
    curr_epoch: int
    curr_batch: int

    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, mf: ModelFactory):
        self.train_loader = DataLoader(train_dataset, batch_size=mf.hp.train_batch_size, shuffle=True, num_workers=3)
        self.val_loader = DataLoader(val_dataset, batch_size=mf.hp.val_batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=mf.hp.test_batch_size, shuffle=False, num_workers=0)
        self.mf = mf
        self.curr_epoch = 1
        self.curr_batch = 1
        self.log_service = LogService()
        self.model = mf.create_pearl_model()
        self.optimizer = torch.optim.AdamW(self.model.get_param_groups(), lr=mf.hp.learning_rate, weight_decay=mf.hp.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)
        self.criterion = nn.L1Loss(reduction="mean")
        self.metric = nn.L1Loss(reduction="sum")


    def train(self):
        print(self.seed, "SEED")
        self.logger.info("Configuration:\n" + OmegaConf.to_yaml(self.mf.hp))


    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        for self.curr_batch, batch in enumerate(self.train_loader, 1):
            total_loss += self.train_batch(batch)

        return total_loss / len(self.train_loader.dataset)


    def train_batch(self, batch: Batch) -> float:
        self.optimizer.zero_grad()
        y_pred = self.model(batch)               # [B]
        loss = self.criterion(y_pred, batch.y)   # [1]
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        self.scheduler.step()

        return loss * batch.y.size(0)


    def lr_lambda(self, curr_step: int) -> float:
        """
        Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/optimization.py#L79
        """
        if curr_step < self.mf.hp.n_warmup_steps:
            return curr_step / max(1, self.mf.hp.n_warmup_steps)
        else:
            return max(0.0, (self.n_total_steps - curr_step) / max(1, self.n_total_steps - self.mf.hp.n_warmup_steps))


    @property
    def n_total_steps(self) -> int:
        return len(self.train_loader) * self.mf.hp.num_epochs


