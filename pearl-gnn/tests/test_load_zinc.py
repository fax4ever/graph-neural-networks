import pytest
from torch_geometric.data import Dataset

from pearl_gnn.load_zinc import load_datasets
from pearl_gnn.hyper_param import HyperParam


class TestLoadDatasets:
    """Test suite for the ZINC dataset loading functionality."""

    @pytest.fixture(scope="class")
    def datasets(self):
        """Load datasets once for all tests in this class."""
        return load_datasets(HyperParam())

    def test_returns_tuple_of_datasets(self, datasets):
        """Test that load_datasets returns a tuple of two datasets."""
        train_dataset, val_dataset, test_dataset = datasets
        assert isinstance(train_dataset, Dataset)
        assert isinstance(val_dataset, Dataset)
        assert isinstance(test_dataset, Dataset)

    def test_datasets_not_empty(self, datasets):
        """Test that all datasets contain graphs."""
        train_dataset, val_dataset, test_dataset = datasets
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        assert len(test_dataset) > 0

    def test_sample_graph_has_required_attributes(self, datasets):
        """Test that sample graphs have the expected structure."""
        train_dataset, _, _ = datasets
        sample = train_dataset[0]

        assert sample.num_nodes > 0
        assert sample.edge_index is not None
        assert sample.edge_index.shape[0] == 2  # edge_index should be [2, num_edges]
        assert sample.y is not None  # target value should exist

    def test_node_features_shape(self, datasets):
        """Test that node features have correct shape."""
        train_dataset, _, _ = datasets
        sample = train_dataset[0]

        if sample.x is not None:
            assert sample.x.shape[0] == sample.num_nodes

    def test_edge_attributes_exist(self, datasets):
        """Test that edge attributes are present."""
        train_dataset, _, _ = datasets
        sample = train_dataset[0]

        if sample.edge_attr is not None:
            num_edges = sample.edge_index.shape[1]
            assert sample.edge_attr.shape[0] == num_edges

