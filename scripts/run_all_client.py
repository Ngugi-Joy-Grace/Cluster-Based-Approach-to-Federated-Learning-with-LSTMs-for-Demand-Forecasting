import subprocess
import os
import time
import logging
from typing import List
from pathlib import Path
from datetime import datetime

def setup_logging() -> None:
    """Configure logging with a timestamped log file."""
    log_filename = datetime.now().strftime('client_logs_%Y%m%d_%H%M%S.log')
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def launch_client(store_id: int, cluster_id: int) -> subprocess.Popen:
    try:
        command = ["python", "run_client.py", str(store_id), str(cluster_id)]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        logging.info(f"Launched client {store_id} for cluster {cluster_id}")
        return proc
    except subprocess.SubprocessError as e:
        logging.error(f"Failed to launch client {store_id} for cluster {cluster_id}: {e}")
        return None

def wait_for_processes(processes: List[subprocess.Popen]) -> None:
    for proc in processes:
        if proc is None:
            continue
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            logging.error(f"Process failed with return code {proc.returncode}")
            if stderr:
                logging.error(f"Error output: {stderr}")

def main():
    setup_logging()

    # Correctly defined federated data directory path
    FEDERATED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "federated_data"
    MAX_CLIENTS_AT_ONCE = 50
    active_processes: List[subprocess.Popen] = []

    if not FEDERATED_DATA_DIR.exists():
        logging.error(f"Federated data directory not found: {FEDERATED_DATA_DIR}")
        return

    total_clients = 0

    for cluster_folder in FEDERATED_DATA_DIR.iterdir():
        if not cluster_folder.is_dir():
            continue
        cluster_id = int(cluster_folder.name.split("_")[1])

        for store_file in cluster_folder.iterdir():
            if not store_file.name.startswith("store_"):
                continue
            store_id = int(store_file.stem.replace("store_", ""))

            proc = launch_client(store_id, cluster_id)
            if proc:
                active_processes.append(proc)
                total_clients += 1

            if len(active_processes) >= MAX_CLIENTS_AT_ONCE:
                logging.info(f"Waiting for batch of {len(active_processes)} clients to complete...")
                wait_for_processes(active_processes)
                active_processes = []

            time.sleep(0.1)

    if active_processes:
        logging.info(f"Waiting for final {len(active_processes)} clients to complete...")
        wait_for_processes(active_processes)

    logging.info(f"Successfully launched and completed {total_clients} client processes")

if __name__ == "__main__":
    main()
