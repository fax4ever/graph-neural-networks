from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.model.mlp import MLP


class ModelFactory:
    def __init__(self, hp: HyperParam):
        self.hp = hp

    def build_model(self):
        pass

    def create_mlp(self, in_dims: int, out_dims: int) -> MLP:
        return MLP(in_dims, out_dims, self.hp)