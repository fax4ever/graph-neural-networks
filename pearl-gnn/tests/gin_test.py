import torch
from pearl_gnn.model.gin import GIN, GINLayer
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.hyper_param import HyperParam


def make_hp():
    """Create HyperParam with matching dims for residual connections."""
    hp = HyperParam()
    hp.pearl_mlp_out = 40
    hp.sample_aggr_hidden_dims = 40
    hp.pe_dims = 40
    return hp


class TestGIN:
    def test_gin_layer_forward(self):
        """Test GINLayer with a tiny graph."""
        mf = ModelFactory(make_hp())
        layer = GINLayer(mf, in_dims=40, out_dims=40)
        
        # 4 nodes, 4 edges (simple cycle)
        X = torch.randn(4, 40)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        
        out = layer(X, edge_index)
        assert out.shape == (4, 40)

    def test_gin_forward(self):
        """Test full GIN model with a tiny graph."""
        hp = make_hp()
        mf = ModelFactory(hp)
        gin = GIN(mf)
        
        # 4 nodes, 4 edges (simple cycle)
        X = torch.randn(4, hp.pearl_mlp_out)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        
        out = gin(X, edge_index)
        assert out.shape == (4, hp.pe_dims)
