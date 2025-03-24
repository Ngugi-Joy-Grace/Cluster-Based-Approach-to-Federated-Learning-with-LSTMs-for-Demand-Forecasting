import subprocess
import logging
import time
from typing import List
from pathlib import Path

def launch_client(cluster_id: int, store_id: int) -> subprocess.Popen:
    """
    Launch a single client process for the given cluster_id and store_id.
    """
    try:
        cmd = ["python", "run_client.py", str(cluster_id), str(store_id)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logging.info(f"Launched client for Cluster={cluster_id}, Store={store_id}")
        return proc
    except subprocess.SubprocessError as e:
        logging.error(f"Failed to launch client for Cluster={cluster_id}, Store={store_id}: {e}")
        return None

def wait_for_processes(processes: List[subprocess.Popen]) -> None:
    """Wait for processes to complete, logging any errors."""
    for proc in processes:
        if proc is None:
            continue
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            logging.error(f"Process failed with return code {proc.returncode}")
            if stderr:
                logging.error(f"Error output: {stderr}")

def main():
    logging.basicConfig(level=logging.INFO)

    FEDERATED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "federated_data"
    if not FEDERATED_DATA_DIR.exists():
        logging.error(f"Federated data directory not found at {FEDERATED_DATA_DIR}")
        return

    MAX_CONCURRENT_CLIENTS = 50
    active_processes: List[subprocess.Popen] = []
    total_launched = 0

    # Iterate over cluster folders
    for cluster_folder in sorted(FEDERATED_DATA_DIR.glob("cluster_*")):
        if not cluster_folder.is_dir():
            continue

        # Parse cluster_id from folder name "cluster_#"
        try:
            cluster_id = int(cluster_folder.name.split("_")[1])
        except ValueError:
            logging.warning(f"Skipping folder {cluster_folder}, cannot parse cluster ID.")
            continue

        # For each .pkl file: "store_{store_id}.pkl"
        for data_file in sorted(cluster_folder.glob("store_*.pkl")):
            store_id_str = data_file.stem.split("_")[1]  # store_123 -> "123"
            store_id = int(store_id_str)

            # Launch one client
            proc = launch_client(cluster_id, store_id)
            if proc:
                active_processes.append(proc)
                total_launched += 1

            # Throttle concurrency
            if len(active_processes) >= MAX_CONCURRENT_CLIENTS:
                logging.info(f"Waiting for {len(active_processes)} clients to finish...")
                wait_for_processes(active_processes)
                active_processes.clear()

            # Small pause to avoid overwhelming the system
            time.sleep(0.1)

    # Wait for any leftover processes
    if active_processes:
        logging.info(f"Waiting for the last {len(active_processes)} clients to finish...")
        wait_for_processes(active_processes)
        active_processes.clear()

    logging.info(f"All done! Total clients launched: {total_launched}")

if __name__ == "__main__":
    main()
