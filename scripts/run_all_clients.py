import math
import subprocess
import logging
import threading
import time
import random
from typing import List
from pathlib import Path
from configs.logging_config import get_run_timestamp_dir, clear_existing_handlers

# Constants
MAX_CONCURRENT_CLIENTS = 10         # Launch 10 at a time
FIXED_SAMPLE_SIZE = 30           # Sample 30 stores per cluster

def stream_process_output(proc: subprocess.Popen, name: str):
    """Continuously read lines from proc.stdout and log them in real time (non-blocking)."""
    if proc.stdout is None:
        return
    for line in proc.stdout:
        logging.info(f"[{name}] {line.rstrip()}")

def launch_client(cluster_id: int, store_id: int, log_file_path: str) -> subprocess.Popen:
    """Launch a single client process (multi-round)."""
    cmd = [
        "python", "run_client.py",
        str(cluster_id),
        str(store_id),
        log_file_path
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        logging.info(f"Launched client for cluster={cluster_id}, store={store_id}")

        # Stream output in the background
        t = threading.Thread(target=stream_process_output, args=(proc, f"Client-{cluster_id}-{store_id}"), daemon=True)
        t.start()
        return proc
    except subprocess.SubprocessError as e:
        logging.error(f"Failed to launch client for cluster={cluster_id}, store={store_id}: {e}")
        return None

def main():
    # ----------------------------------------------------------------------
    # 1) Clear existing handlers to avoid stacking them if run multiple times
    clear_existing_handlers()

    # 2) Set up a single logging configuration for the entire script
    run_dir = get_run_timestamp_dir()  # uses the environment-based timestamp directory
    log_file_path = run_dir / "run_all_clients.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    # ----------------------------------------------------------------------

    FEDERATED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "federated_data_scaled"
    if not FEDERATED_DATA_DIR.exists():
        logging.error(f"Federated data directory not found: {FEDERATED_DATA_DIR}")
        return

    active_processes: List[subprocess.Popen] = []
    total_launched = 0

    # Iterate over cluster_* folders
    for cluster_folder in sorted(FEDERATED_DATA_DIR.glob("cluster_*")):
        if not cluster_folder.is_dir():
            continue

        try:
            cluster_id = int(cluster_folder.name.split("_")[1])
        except ValueError:
            logging.warning(f"Skipping folder {cluster_folder}, cannot parse cluster ID.")
            continue

        logging.info(f"=== Processing Cluster {cluster_id} ===")

        # Each cluster can have its own client log file name (passed to run_client.py)
        cluster_log_filename = f"clients_cluster_{cluster_id}.log"
        store_files = sorted(cluster_folder.glob("store_*.pkl"))

        num_total = len(store_files)
        if num_total == 0:
            logging.info(f"No store files found for cluster {cluster_id}, skipping.")
            continue

        # 1) Calculate sample_count at 5%
        sample_count = min(FIXED_SAMPLE_SIZE, num_total)
        logging.info(f"Cluster {cluster_id} has {num_total} stores in total.")
        logging.info(f"Fixed sample size => {sample_count} stores to be launched for cluster {cluster_id}.")

        # 2) Actually pick the subset
        if sample_count < num_total:
            selected_files = random.sample(store_files, sample_count)
        else:
            selected_files = store_files

        # 3) Calculate how many 'batches' for concurrency=10 (just for logging)
        batches = math.ceil(sample_count / MAX_CONCURRENT_CLIENTS)
        logging.info(
            f"Cluster {cluster_id} => concurrency={MAX_CONCURRENT_CLIENTS}, "
            f"sample_count={sample_count}, => ~{batches} batch(es)."
        )

        launched_count_for_cluster = 0

        # 4) Launch the sampled files in sets of 10
        for i, data_file in enumerate(selected_files, start=1):
            store_id = int(data_file.stem.replace("store_", ""))

            proc = launch_client(cluster_id, store_id, cluster_log_filename)
            if proc:
                active_processes.append(proc)
                total_launched += 1
                launched_count_for_cluster += 1

            # Enforce concurrency limit
            while len(active_processes) >= MAX_CONCURRENT_CLIENTS:
                logging.info(f"At concurrency limit ({MAX_CONCURRENT_CLIENTS}). Waiting for some to free up.")
                time.sleep(2)
                # Remove finished processes from active_processes
                active_processes = [p for p in active_processes if p.poll() is None]

            # Optional small delay
            time.sleep(0.1)

            if (i % MAX_CONCURRENT_CLIENTS) == 0 or i == sample_count:
                logging.info(f"[Cluster {cluster_id}] Launched {i}/{sample_count} stores so far...")

        logging.info(f"[Cluster {cluster_id}] Done launching all {launched_count_for_cluster} sampled stores.")

    # After all clusters
    logging.info(f"All clusters done. Total clients launched across clusters: {total_launched}")
    logging.info("Waiting for client processes to remain alive until the server finishes FL rounds...")

    # Final wait until all client processes exit
    while any(p.poll() is None for p in active_processes):
        still_running = sum(p.poll() is None for p in active_processes)
        logging.info(f"{still_running} clients still running...")
        time.sleep(5)
        active_processes = [p for p in active_processes if p.poll() is None]

    logging.info("All done! All sampled clients have exited (the server ended).")

if __name__ == "__main__":
    main()

