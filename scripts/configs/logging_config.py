import logging
from pathlib import Path
from datetime import datetime
import os


def get_run_timestamp_dir() -> Path:
    """
    Creates and returns a timestamped directory for the current run.
    Uses an environment variable to ensure the same directory is used within a run.
    
    Returns:
        Path: Directory path for the current run's logs
    """
    # Check if we already have a run directory from environment
    run_timestamp = os.environ.get('FEDERATION_RUN_TIMESTAMP')

    if run_timestamp is None:
        # Create new timestamp if this is the first call
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ['FEDERATION_RUN_TIMESTAMP'] = run_timestamp

    logs_dir = Path("../logs")
    run_dir = logs_dir / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class ClusterFilter(logging.Filter):
    """Inject the cluster_id into the log record so '%(cluster_id)d' works in the format."""

    def __init__(self, cluster_id: int):
        super().__init__()
        self.cluster_id = cluster_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.cluster_id = self.cluster_id
        return True


class DeprecatedFeatureFilter(logging.Filter):
    """
    Filter out lines containing the phrase 'DEPRECATED FEATURE'.
    This is useful to hide Flower's deprecation warnings in logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if "DEPRECATED FEATURE" in record.getMessage():
            return False
        return True


def clear_existing_handlers():
    """Explicitly remove existing logging handlers to ensure proper reconfiguration."""
    root_logger = logging.getLogger()
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])


def setup_server_logging(cluster_id: int, remove_deprecated_feature_logs: bool = True):
    """
    Configure logging for a server. Each cluster server logs to 'server_cluster_{cluster_id}.log'
    within a timestamped run directory. By default, we remove lines containing 'DEPRECATED FEATURE'.
    """
    run_dir = get_run_timestamp_dir()
    log_filename = run_dir / f"server_cluster_{cluster_id}.log"

    # Clear existing handlers explicitly before configuring
    clear_existing_handlers()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [Server %(cluster_id)d] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    # Attach filters to all handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(ClusterFilter(cluster_id))
        if remove_deprecated_feature_logs:
            handler.addFilter(DeprecatedFeatureFilter())

    logging.info(f"[Server {cluster_id}] Logging initialized -> {log_filename}")


def setup_clients_logging(log_file_path: str, cluster_id: int, remove_deprecated_feature_logs: bool = True):
    """
    Configure logging for clients within a timestamped run directory.
    The 'log_file_path' is the base filename, placed inside the run directory.

    By default, we also remove lines containing 'DEPRECATED FEATURE'.
    """
    run_dir = get_run_timestamp_dir()

    # Extract just the filename from the path and place it in the run directory
    log_filename = Path(log_file_path).name
    full_path = run_dir / log_filename

    # Clear existing handlers explicitly before configuring
    clear_existing_handlers()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [Cluster %(cluster_id)d - CLIENT] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(full_path),
            logging.StreamHandler()
        ]
    )

    # Attach filters to all handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(ClusterFilter(cluster_id))
        if remove_deprecated_feature_logs:
            handler.addFilter(DeprecatedFeatureFilter())

    logging.info(f"Initialized client logging for cluster {cluster_id} -> {full_path}")
