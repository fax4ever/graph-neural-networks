from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.dataset.load_zinc import load_datasets
from pearl_gnn.helper.visualize import plot

if __name__ == "__main__":  
    hyper_param = HyperParam()
    
    train_dataset, val_dataset, test_dataset = load_datasets(hyper_param)
    for i in range(3):
        plot(train_dataset[i], save_path=f"graph_zinc_train_{i}.png")
        plot(val_dataset[i], save_path=f"graph_zinc_val_{i}.png")
        plot(test_dataset[i], save_path=f"graph_zinc_test_{i}.png")