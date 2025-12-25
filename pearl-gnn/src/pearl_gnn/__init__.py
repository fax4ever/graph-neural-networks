from pearl_gnn.client import MyClient
from pearl_gnn.load_dataset import load_datasets


def main() -> None:
    client = MyClient()
    client.info()

    load_datasets()

    

