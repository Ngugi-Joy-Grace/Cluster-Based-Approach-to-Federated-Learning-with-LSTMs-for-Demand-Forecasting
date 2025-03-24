import logging
from pathlib import Path
from datetime import datetime

def setup_server_logging(cluster_id: int):
    """
    Configure logging for a server. Each cluster server logs to 'server_cluster_{cluster_id}.log'.
    Add a timestamp if you prefer unique filenames per run.
    """
    logs_dir = Path("../logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"server_cluster_{cluster_id}_{timestamp}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - [Server %(cluster_id)d] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # Because we used '%(cluster_id)d' in the format, we must supply it via extra or by changing the format.
    # Easiest fix: remove '%(cluster_id)d' in the format, or we do an override below:
    for handler in logging.getLogger().handlers:
        handler.addFilter(_ClusterFilter(cluster_id))

    logging.info(f"[Server {cluster_id}] Logging initialized -> {log_filename}")


def setup_clients_logging(log_file_path: str, cluster_id: int):
    logs_dir = Path("../logs")
    logs_dir.mkdir(exist_ok=True)

    full_path = logs_dir / log_file_path  # e.g. "../logs/clients_cluster_0.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - [Cluster %(cluster_id)d - CLIENT] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(full_path),
            logging.StreamHandler()
        ]
    )

    for h in logging.getLogger().handlers:
        h.addFilter(_ClusterFilter(cluster_id))

    logging.info(f"Initialized client logging for cluster {cluster_id} -> {full_path}")

class _ClusterFilter(logging.Filter):
    def __init__(self, cluster_id):
        super().__init__()
        self.cluster_id = cluster_id

    def filter(self, record):
        record.cluster_id = self.cluster_id
        return True
