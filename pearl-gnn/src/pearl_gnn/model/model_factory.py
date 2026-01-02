from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.model.mlp import MLP
from pearl_gnn.model.pearl import PEARL_GNN_Model


class ModelFactory:
    def __init__(self, hp: HyperParam):
        self.hp = hp

    def create_mlp(self, in_dims: int, out_dims: int) -> MLP:
        return MLP(in_dims, out_dims, self.hp)

    def create_pearl_model(self) -> PEARL_GNN_Model:
        return PEARL_GNN_Model(self)