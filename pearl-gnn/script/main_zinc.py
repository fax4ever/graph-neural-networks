from pearl_gnn.client import MyClient
from pearl_gnn.load_zinc import load_datasets
from pearl_gnn.visualize import plot

if __name__ == "__main__":
    client = MyClient()
    client.info()
    
    train_dataset, test_dataset = load_datasets()
    for i in range(10):
        plot(train_dataset[i], save_path=f"graph_{i}.png")
