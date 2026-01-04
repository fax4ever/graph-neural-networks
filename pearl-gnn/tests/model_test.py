from pearl_gnn.model.pearl import PEARL_GNN_Model
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.hyper_param import HyperParam


class TestModel:
    def test_model_creation(self):
        model = PEARL_GNN_Model(ModelFactory(HyperParam()))
        assert model is not None