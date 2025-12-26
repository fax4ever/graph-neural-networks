from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.load_zinc import load_datasets
from pearl_gnn.visualize import plot

if __name__ == "__main__":  
    hyper_param = HyperParam()
    hyper_param.set_seed()
    
    train_dataset, val_dataset, test_dataset = load_datasets()
    for i in range(10):
        plot(train_dataset[i], save_path=f"graph_zinc_{i}.png")