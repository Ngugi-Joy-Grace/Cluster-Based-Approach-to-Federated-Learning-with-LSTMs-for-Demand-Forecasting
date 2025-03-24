import logging
from pathlib import Path
from datetime import datetime

def get_run_timestamp_dir() -> Path:
    """
    Creates and returns a timestamped directory for the current run.
    
    Returns:
        Path: Directory path for the current run's logs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("../logs")
    run_dir = logs_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def setup_server_logging(cluster_id: int):
    """
    Configure logging for a server. Each cluster server logs to 'server_cluster_{cluster_id}.log'
    within a timestamped run directory.
    """
    run_dir = get_run_timestamp_dir()
    log_filename = run_dir / f"server_cluster_{cluster_id}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - [Server %(cluster_id)d] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    for handler in logging.getLogger().handlers:
        handler.addFilter(_ClusterFilter(cluster_id))

    logging.info(f"[Server {cluster_id}] Logging initialized -> {log_filename}")

def setup_clients_logging(log_file_path: str, cluster_id: int):
    """
    Configure logging for clients within a timestamped run directory.
    """
    run_dir = get_run_timestamp_dir()
    
    # Extract just the filename from the path and place it in the run directory
    log_filename = Path(log_file_path).name
    full_path = run_dir / log_filename

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
