import flwr as fl
from configs.logging_config import setup_server_logging

if __name__ == "__main__":
    cluster_id = 0
    setup_server_logging(cluster_id)
    # Define federated averaging strategy
    strategy = fl.server.strategy.FedAvg(
        # Adjust these as needed
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
