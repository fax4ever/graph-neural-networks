from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.datasets import ZINC
from torch_geometric.utils import get_laplacian, to_dense_adj
from pathlib import Path
from pearl_gnn.model.pearl import PEARL_GNN_Model
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.model.model_support import ModelSupport
from pearl_gnn.model.pe import add_laplacian_transform, get_per_graph_dense_laplacians


ROOT = str(Path(__file__).parent.parent / "data" / "ZINC")
SUB_DATASET_SIZE = 10
hyper_param = HyperParam()
N_TOTAL_STEPS = SUB_DATASET_SIZE * hyper_param.num_epochs


def lr_lambda(curr_step: int) -> float:
    """
    Based on https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/optimization.py#L79
    """
    if curr_step < hyper_param.n_warmup_steps:
        return curr_step / max(1, hyper_param.n_warmup_steps)
    else:
        return max(0.0, (N_TOTAL_STEPS - curr_step) / max(1, N_TOTAL_STEPS - hyper_param.n_warmup_steps))


class TestModel:
    def test_model_creation(self):
        model = PEARL_GNN_Model(ModelFactory(HyperParam()))
        assert model is not None


    def test_laplacian_transform_single_graph(self):
        """Test that the transform correctly adds sparse Laplacian components."""
        train_dataset: Dataset = ZINC(root=ROOT, subset=True, split="train")
        data = train_dataset[0]
        
        # Apply the transform
        data = add_laplacian_transform(data)
        
        assert hasattr(data, 'lap_edge_index'), "Transform should add lap_edge_index"
        assert hasattr(data, 'lap_edge_attr'), "Transform should add lap_edge_attr"
        assert data.lap_edge_index.shape[0] == 2, "lap_edge_index should have 2 rows"


    def test_laplacian_transform_with_dataloader(self):
        """Test that transformed data can be properly batched by PyG."""
        train_dataset: Dataset = ZINC(root=ROOT, subset=True, split="train", transform=add_laplacian_transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        for batch in train_loader:
            assert hasattr(batch, 'lap_edge_index'), "Batch should have lap_edge_index"
            assert hasattr(batch, 'lap_edge_attr'), "Batch should have lap_edge_attr"
            assert hasattr(batch, '_slice_dict'), "Batch should have slice info"
            assert 'lap_edge_index' in batch._slice_dict, "Slice dict should contain lap_edge_index"
            break  # Just test first batch


    def test_per_graph_laplacian_reconstruction(self):
        """Test that we can reconstruct per-graph dense Laplacians from a batch."""
        train_dataset: Dataset = ZINC(root=ROOT, subset=True, split="train", transform=add_laplacian_transform)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
        
        for batch in train_loader:
            # Reconstruct per-graph Laplacians
            laplacians = get_per_graph_dense_laplacians(batch)
            
            # Verify we get one Laplacian per graph
            assert len(laplacians) == batch.num_graphs, f"Expected {batch.num_graphs} Laplacians"
            
            # Verify each Laplacian has correct shape
            ptr = batch.ptr
            for i, lap in enumerate(laplacians):
                n_nodes = (ptr[i + 1] - ptr[i]).item()
                assert lap.shape == (n_nodes, n_nodes), f"Laplacian {i} has wrong shape"
            break  # Just test first batch


    def test_laplacian_values_correctness(self):
        """Verify reconstructed Laplacians match directly computed ones."""
        train_dataset: Dataset = ZINC(root=ROOT, subset=True, split="train", transform=add_laplacian_transform)
        
        # Get a single graph
        data = train_dataset[0]
        
        # Compute Laplacian directly
        n = data.num_nodes
        L_edge_index, L_values = get_laplacian(data.edge_index, normalization="sym", num_nodes=n)
        direct_lap = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(0)
        
        # Create a "batch" of one graph and reconstruct
        batch = Batch.from_data_list([data])
        reconstructed_laps = get_per_graph_dense_laplacians(batch)
        
        import torch
        assert torch.allclose(direct_lap, reconstructed_laps[0], atol=1e-6), \
            "Reconstructed Laplacian should match directly computed one"


    def test_subdataset_training(self):
        """Test that training works with the new Laplacian handling."""
        dataset: Dataset = ZINC(root=ROOT, subset=True, split="train", transform=add_laplacian_transform)
        dataset = dataset[:SUB_DATASET_SIZE]
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model_factory = ModelFactory(hyper_param)
        model = ModelSupport(model_factory, lr_lambda)

        model.train_epoch(loader)

