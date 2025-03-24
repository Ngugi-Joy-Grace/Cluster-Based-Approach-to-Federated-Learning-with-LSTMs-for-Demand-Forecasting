import flwr as fl
import numpy as np
import pandas as pd
import sys
from client import StoreClient

def load_store_data(store_id, cluster_id):
    """
    Loads and prepares the dataset for a given store from stored CSV files.

    Args:
        store_id (int): ID of the store.
        cluster_id (int): ID of the cluster.

    Returns:
        Tuple: Training and validation datasets.
    """
    file_path = f'../data/federated_data/cluster_{cluster_id}/store_{store_id}.csv'
    store_df = pd.read_csv(file_path)

    X = store_df.drop(['Sales', 'Date', 'Store', 'Cluster'], axis=1).values
    y = store_df['Sales'].values
    X = X.reshape(X.shape[0], 1, X.shape[1])

    split_idx = int(len(X) * 0.8)
    return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:])

if __name__ == "__main__":
    store_id = int(sys.argv[1])
    cluster_id = int(sys.argv[2])

    train_data, val_data = load_store_data(store_id, cluster_id)

    fl.client.start_numpy_client(
        server_address="[::]:8080",
        client=StoreClient(train_data, val_data)
    )
