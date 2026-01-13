import os
from typing import Callable
from pathlib import Path
import logging
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Batch, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.model.pearl import PEARL_GNN_Model


# This code is adapted from:
# https://github.com/ehejin/Pearl-PE/blob/main/PEARL/src/zinc/trainer.py

# @fax4ever: 
# I tried to rewrite the code to be more readable and modular, simplify what is not necessary.
# In this process I could make some mistakes, so please check the code and report any issues.


def plot_epochs_progress(train_losses, val_losses, test_losses, output_dir, file_name):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot train loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train MAE", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Train MAE per Epoch')

    # Plot validation loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_losses, label="Validation MAE", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE per Epoch')

    # Plot test loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_losses, label="Test MAE", color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Test MAE per Epoch')

    # Save plots in the current directory
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


class ModelTrainingData:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []


    def append(self, train_loss, val_loss, test_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.test_losses.append(test_loss)


    def plot_training(self, model_name, output_dir):
        plot_epochs_progress(self.train_losses, self.val_losses, self.test_losses, output_dir, f"epochs_progress_{model_name}.png")


class ModelSupport:
    def __init__(self, mf: ModelFactory, lr_lambda: Callable[[int], float], model_name: str = "pearl", test_dir_name: str = "baseline"):

        base_dir = Path(__file__).parent.parent.parent.parent

        self.model = PEARL_GNN_Model(mf)
        self.device = mf.hp.device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.get_param_groups(), lr=mf.hp.learning_rate, weight_decay=mf.hp.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        # Use mean absolute error (MAE) as in the original paper
        self.criterion = nn.L1Loss(reduce="mean") 
        self.metric = nn.L1Loss(reduction="sum")
        self.model_name = model_name

        checkpoint_dir = os.path.join(base_dir, "checkpoints", f"{model_name}_{test_dir_name}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_path = os.path.join(checkpoint_dir, f"model_{test_dir_name}")
        self.best_checkpoint_path = os.path.join(base_dir, "checkpoints", f"{model_name}_{test_dir_name}_best.pth")
        self.logs_path = os.path.join(base_dir, "logs", test_dir_name)
        os.makedirs(self.logs_path, exist_ok=True)
        self.training_data = ModelTrainingData()
        # We select the model that has the lowest validation loss as in the original paper
        self.best_val_loss = 999.0 # 999.0 is used in the orginal paper


    def parameters(self):
        return self.model.parameters()


    def train_epoch(self, train_loader: DataLoader, epoch: int, num_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for self.curr_batch, batch in enumerate(pbar, 1):
            batch_loss, batch_prediction = self.train_batch(batch)
            total_loss += batch_loss
            total_correct += (batch_prediction ==  batch.y).sum().item()
            pbar.set_postfix(loss=f"{total_loss / (self.curr_batch * train_loader.batch_size):.4f}")

        size = len(train_loader.dataset)
        return total_loss / size, total_correct / size


    def evaluate_epoch(self, eval_loader: DataLoader, desc: str = "Eval") -> float:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0

        pbar = tqdm(eval_loader, desc=f"         [{desc}]", leave=False)
        for self.curr_batch, batch in enumerate(pbar, 1):
            batch_loss, batch_prediction = self.evaluate_batch(batch)
            total_loss += batch_loss
            total_correct += (batch_prediction ==  batch.y).sum().item()

        size = len(eval_loader.dataset)
        return total_loss / size, total_correct / size     


    def train_batch(self, batch: Batch):
        batch.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(batch)
        loss = self.criterion(output, batch.y)
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        self.scheduler.step()

        # @fax4ever:
        # multiplying the loss by the batch size I think we make the criterion
        # consistent with the metric reduction (sum)
        return loss * batch.y.size(0), output


    def evaluate_batch(self, batch: Batch) -> float:
        batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch)
        return self.metric(output, batch.y).item(), output 


    def save_checkpoint(self, current_epoch):
        checkpoint_file = f"{self.checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(self.model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")


    def append_epoch_data(self, train_loss, val_loss, test_loss, epoch, num_epochs):
        self.training_data.append(train_loss, val_loss, test_loss)

        info = f"Epoch {epoch + 1}/{num_epochs}, Train MAE: {train_loss:.4f}, " \
               f"Val MAE: {val_loss:.4f}, Test MAE: {test_loss:.4f}"
        print(info)
        logging.info(info)

        # Save best model according to the validation set (lower is better for MAE)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.best_checkpoint_path)
            print(f"  -> Best model updated! (Val MAE: {val_loss:.4f})")


    def load_best_checkpoint(self):
        self.model.load_state_dict(torch.load(self.best_checkpoint_path))


    def plot_training(self):
        # Plot training progress in current directory
        self.training_data.plot_training(self.model_name, self.logs_path)    