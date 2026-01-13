import torch
import random
import numpy as np


class HyperParam:
    # Dataset properties
    # 28 / 3 is for ZINC
    # those discrete values will be used as input for
    # embedding layers
    n_node_types: int = 28
    n_edge_types: int = 3

    # General hyperparameters
    seed: int = 42
    device: torch.device
    device_name: str

    # Model > MLP hyperparameters
    n_mlp_layers: int = 3
    mlp_hidden_dims: int = 128
    mlp_dropout_prob: float = 0.0
    mlp_norm_type: str = "batch"

    # Model > GINE hyperparameters
    n_base_layers: int = 4
    node_emb_dims: int = 128
    base_hidden_dims: int = 128
    gine_model_bn: bool = False
    pooling: str = "add"
    target_dim: int = 1

    # Model > GIN / SampleAggregator hyperparameters
    gin_sample_aggregator_bn: bool = True
    n_sample_aggr_layers: int = 8
    sample_aggr_hidden_dims: int = 40

    # Model > Positional Encoding / PEARL
    pe_dims: int = 37 # based on SPE paper by Huang et al. (2023)
    basis: bool = True # True for B-PEARL, False for R-PEARL
    num_samples = 120 # num_samples for R-PEARL (used only if basis is false!)
    pearl_mlp_out = 37
    pearl_k: int = 7
    pearl_mlp_nlayers: int = 1
    pearl_mlp_hid: int = 37
    pearl_mlp_out: int = 37

    # Dataset hyperparameters
    use_subset: bool = True
    train_batch_size: int = 32
    val_batch_size: int = 32
    test_batch_size: int = 32
    
    # Training hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    num_epochs: int = 2 #1400 in the original paper!
    n_warmup_steps: int = 100
    
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_name = "MPS"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"
        self.set_seed()    


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