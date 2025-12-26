import torch
import random
import numpy as np


class HyperParam:
    def __init__(self):
        self.seed = 42
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


    def set_seed(self) -> None:
        """
        Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_utils.py#L83
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)              


    def device(self) -> torch.device:
        return self.device