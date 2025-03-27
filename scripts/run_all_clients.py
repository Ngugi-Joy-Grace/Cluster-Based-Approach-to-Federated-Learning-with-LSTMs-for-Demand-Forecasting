import subprocess
import logging
import threading
import time
from typing import List
from pathlib import Path
from configs.logging_config import get_run_timestamp_dir


def stream_process_output(proc: subprocess.Popen, name: str):
    if proc.stdout is None:
        return
    for line in proc.stdout:
        logging.info(f"[{name}] {line.rstrip()}")


def launch_client(cluster_id: int, store_id: int, log_file_path: str) -> subprocess.Popen:
    """
    Launch a single client process for the given cluster_id and store_id.
    """
    try:
        cmd = [
            "python", "run_client.py",
            str(cluster_id),
            str(store_id),
            log_file_path
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        logging.info(f"Launched client for Cluster={cluster_id}, Store={store_id}")
        # Stream output in a background thread, non-blocking
        t = threading.Thread(target=stream_process_output, args=(proc, f"Client-{cluster_id}-{store_id}"), daemon=True)
        t.start()

        return proc
    except subprocess.SubprocessError as e:
        logging.error(f"Failed to launch client for Cluster={cluster_id}, Store={store_id}: {e}")
        return None



def clear_existing_handlers():
    root_logger = logging.getLogger()
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])


def main():
    # Get the run directory for this execution
    run_dir = get_run_timestamp_dir()

    clear_existing_handlers()

    # Setup logging for the run_all_client script itself
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "run_all_clients.log"),
            logging.StreamHandler()
        ]
    )

    FEDERATED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "federated_data"
    if not FEDERATED_DATA_DIR.exists():
        logging.error(f"Federated data directory not found at {FEDERATED_DATA_DIR}")
        return

    MAX_CONCURRENT_CLIENTS = 15
    active_processes: List[subprocess.Popen] = []
    total_launched = 0

    # Iterate over cluster folders
    for cluster_folder in sorted(FEDERATED_DATA_DIR.glob("cluster_*")):
        if not cluster_folder.is_dir():
            continue

        try:
            cluster_id = int(cluster_folder.name.split("_")[1])
        except ValueError:
            logging.warning(f"Skipping folder {cluster_folder}, cannot parse cluster ID.")
            continue

        # Create log filename for this cluster's clients
        cluster_log_filename = f"clients_cluster_{cluster_id}.log"
        # The actual path will be determined by the logging config

        # For each store in this cluster
        for data_file in sorted(cluster_folder.glob("store_*.pkl")):
            store_id = int(data_file.stem.split("_")[1])

            # Launch a client for (cluster_id, store_id)
            proc = launch_client(cluster_id, store_id, cluster_log_filename)
            if proc:
                active_processes.append(proc)
                total_launched += 1

            # Throttle concurrency
            if len(active_processes) >= MAX_CONCURRENT_CLIENTS:
                logging.info(f"Waiting for {len(active_processes)} clients to finish (concurrency limit).")
                # Wait for them to exit
                # But do a short wait or poll until they finish, rather than blocking for indefinite time.
                while any(p.poll() is None for p in active_processes):
                    still_running = sum(p.poll() is None for p in active_processes)
                    logging.info(f"{still_running} processes still running...")
                    time.sleep(2)
                    # Remove completed
                    active_processes = [p for p in active_processes if p.poll() is None]

        time.sleep(0.1) # small delay between launches

        #  After all stores in all clusters are launched, wait for the last batch to finish
        if active_processes:
            logging.info(f"Waiting for the last {len(active_processes)} clients to finish...")
            while any(p.poll() is None for p in active_processes):
                still_running = sum(p.poll() is None for p in active_processes)
                logging.info(f"{still_running} processes still running in final batch...")
                time.sleep(2)
                # Remove completed
                active_processes = [p for p in active_processes if p.poll() is None]

    logging.info(f"All done! Total clients launched: {total_launched}")


if __name__ == "__main__":
    main()
