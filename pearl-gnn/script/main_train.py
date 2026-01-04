from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.model.model_factory import ModelFactory
from pearl_gnn.trainer import Trainer


if __name__ == "__main__":
    hyper_param = HyperParam()
    model_factory = ModelFactory(hyper_param)
    trainer = Trainer(model_factory)
    trainer.train_all_epochs()
