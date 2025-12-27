from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from pearl_gnn.hyper_param import HyperParam

class Trainer:
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, hp: HyperParam):
        self.train_loader = DataLoader(train_dataset, batch_size=hp.train_batch_size, shuffle=True, num_workers=3)
        self.val_loader = DataLoader(val_dataset, batch_size=hp.val_batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=hp.test_batch_size, shuffle=False, num_workers=0)

