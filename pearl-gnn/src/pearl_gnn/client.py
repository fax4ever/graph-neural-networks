import torch


class MyClient:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


    def info(self):
        print(f"Device: {self.device}")