import sys
import logging
import flwr as fl
from pathlib import Path

from client import StoreClient
from configs.logging_config import setup_clients_logging

if __name__ == "__main__":
    # Check if the correct number of arguments was passed
    if len(sys.argv) != 4:
        print("Usage: python run_client.py <cluster_id> <store_id>")
        sys.exit(1)

    cluster_id = int(sys.argv[1])
    store_id = int(sys.argv[2])
    log_file_path = sys.argv[3]

    # Initialize cluster-level logging for all clients in the same cluster
    setup_clients_logging(log_file_path, cluster_id=cluster_id)

    # Path to the folder containing cluster_{cluster_id}/store_{store_id}.pkl
    FEDERATED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "federated_data"

    # Create a client instance
    client = StoreClient(cluster_id, store_id, FEDERATED_DATA_DIR)

    # Determine which server port to use.
    # For example, cluster 0 -> 8080, cluster 1 -> 8081, cluster 2 -> 8082
    server_port = 8080 + cluster_id
    server_address = f"127.0.0.1:{server_port}"

    logging.info(f"[Store {store_id}, Cluster {cluster_id}] Connecting to {server_address} ...")

    # Modern (non-deprecated) way to start the client
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )
