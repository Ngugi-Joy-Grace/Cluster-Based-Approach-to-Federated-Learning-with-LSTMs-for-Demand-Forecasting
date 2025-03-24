import flwr as fl
from configs.logging_config import setup_server_logging
import flwr as fl
from typing import Dict

def weighted_average_fit_metrics(metrics: Dict[str, float], num_examples: int) -> Dict[str, float]:
    """This function is called for each client after fit to gather metrics.
    You can combine them or keep them individually for a final result."""
    # For example, just pass them through or return them in a structure
    return metrics

def aggregate_fit_metrics_fn(
    results,  # List of tuples (client_parameters, num_examples, fit_metrics)
    failures
):
    # results is a list of (parameters, num_examples, metrics)
    # We'll do a simple weighted average of each metric across clients:
    if not results:
        return {}

    # Sum the product of (client_metric * client_num_examples) / total_examples
    metrics_dict = {}
    total_examples = 0
    for _, num_examples, metrics in results:
        total_examples += num_examples

    for _, num_examples, metrics in results:
        for k, v in metrics.items():
            metrics_dict[k] = metrics_dict.get(k, 0.0) + v * num_examples

    for k in metrics_dict:
        metrics_dict[k] /= total_examples

    return metrics_dict

def aggregate_evaluate_metrics_fn(
    results,  # List of tuples (loss, num_examples, metrics)
    failures
):
    # Similar logic for evaluation metrics
    if not results:
        return {}

    metrics_dict = {}
    total_examples = 0
    for _, num_examples, metrics in results:
        total_examples += num_examples

    for _, num_examples, metrics in results:
        for k, v in metrics.items():
            metrics_dict[k] = metrics_dict.get(k, 0.0) + v * num_examples

    for k in metrics_dict:
        metrics_dict[k] /= total_examples

    return metrics_dict



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
        min_evaluate_clients=2,
        fit_metrics_aggregation_fn = aggregate_fit_metrics_fn,
        evaluate_metrics_aggregation_fn = aggregate_evaluate_metrics_fn
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
