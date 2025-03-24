import flwr as fl
from configs.logging_config import setup_server_logging
from typing import Dict, List, Tuple, Optional

def aggregate_metrics(
    results: List[Tuple[Optional[bytes], int, Dict[str, float]]],
    failures: List[BaseException]
) -> Dict[str, float]:
    """
    Aggregate metrics across all clients using weighted average based on number of examples.
    
    Args:
        results: List of tuples containing (client_parameters, num_examples, metrics)
        failures: List of exceptions from failed clients
        
    Returns:
        Dict containing aggregated metrics
    """
    if not results:
        return {}

    metrics_dict = {}
    total_examples = sum(num_examples for _, num_examples, _ in results)

    # Calculate weighted metrics
    for _, num_examples, metrics in results:
        weight = num_examples / total_examples
        for metric_name, metric_value in metrics.items():
            metrics_dict[metric_name] = (
                metrics_dict.get(metric_name, 0.0) + metric_value * weight
            )

    return metrics_dict

if __name__ == "__main__":
    cluster_id = 0
    setup_server_logging(cluster_id)
    
    # Define federated averaging strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        fit_metrics_aggregation_fn=aggregate_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics
    )

    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
