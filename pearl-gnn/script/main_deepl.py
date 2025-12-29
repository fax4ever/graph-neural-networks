from pearl_gnn.hyper_param import HyperParam
from pearl_gnn.dataset.load_deepl import load_datasets
from pearl_gnn.helper.visualize import plot

if __name__ == "__main__":
    hyper_param = HyperParam()
    
    train_dataset, val_dataset, test_dataset = load_datasets(hyper_param, letter="A")
    for i in range(3):
        plot(train_dataset[i], save_path=f"graph_deepl_train_A_{i}.png")
        plot(val_dataset[i], save_path=f"graph_deepl_val_A_{i}.png")
        plot(test_dataset[i], save_path=f"graph_deepl_test_A_{i}.png")

    train_dataset, val_dataset, test_dataset = load_datasets(hyper_param, letter="B")
    for i in range(3):
        plot(train_dataset[i], save_path=f"graph_deepl_train_B_{i}.png")
        plot(val_dataset[i], save_path=f"graph_deepl_val_B_{i}.png")
        plot(test_dataset[i], save_path=f"graph_deepl_test_B_{i}.png")

    train_dataset, val_dataset, test_dataset = load_datasets(hyper_param, letter="C")
    for i in range(3):
        plot(train_dataset[i], save_path=f"graph_deepl_train_C_{i}.png")
        plot(val_dataset[i], save_path=f"graph_deepl_val_C_{i}.png")
        plot(test_dataset[i], save_path=f"graph_deepl_test_C_{i}.png")

    train_dataset, val_dataset, test_dataset = load_datasets(hyper_param, letter="D")
    for i in range(3):
        plot(train_dataset[i], save_path=f"graph_deepl_train_D_{i}.png")
        plot(val_dataset[i], save_path=f"graph_deepl_val_D_{i}.png")
        plot(test_dataset[i], save_path=f"graph_deepl_test_D_{i}.png")            