import flwr as fl

# Define federated averaging strategy
strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2
)

# Start server to orchestrate federated training
fl.server.start_server(
    server_address="[::]:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy
)
