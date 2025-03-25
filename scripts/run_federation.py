import subprocess
import time
import logging
import os
from configs.logging_config import get_run_timestamp_dir

def start_servers() -> list[subprocess.Popen]:
    """Start all federation servers and return their processes."""
    server_processes = []
    
    for cluster_id in range(3):  # For clusters 0, 1, and 2
        cmd = ["python", f"server_cluster_{cluster_id}.py"]
        try:
            # Pass the environment variable to the subprocess
            env = os.environ.copy()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env  # Pass the environment
            )
            server_processes.append(proc)
            logging.info(f"Started server for cluster {cluster_id}")
            time.sleep(2)  # Give each server time to initialize
        except subprocess.SubprocessError as e:
            logging.error(f"Failed to start server {cluster_id}: {e}")
    
    return server_processes

def start_clients() -> subprocess.Popen:
    """Start the client coordination process."""
    try:
        # Pass the environment variable to the subprocess
        env = os.environ.copy()
        proc = subprocess.Popen(
            ["python", "run_all_clients.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env  # Pass the environment
        )
        logging.info("Started client coordinator")
        return proc
    except subprocess.SubprocessError as e:
        logging.error(f"Failed to start clients: {e}")
        return None

def monitor_processes(processes: list[subprocess.Popen], process_names: list[str]):
    """Monitor processes and log their output."""
    for proc, name in zip(processes, process_names):
        if proc is None:
            continue
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            logging.error(f"{name} failed with return code {proc.returncode}")
            if stderr:
                logging.error(f"{name} error output: {stderr}")
        if stdout:
            logging.info(f"{name} output: {stdout}")

def main():
    # Setup run directory and logging
    run_dir = get_run_timestamp_dir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "federation.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting federation training run")
    
    # Start all servers
    server_processes = start_servers()
    if not server_processes or len(server_processes) != 3:
        logging.error("Failed to start all servers")
        return
    
    # Give servers time to fully initialize
    time.sleep(5)
    
    # Start clients
    client_proc = start_clients()
    if not client_proc:
        logging.error("Failed to start clients")
        for proc in server_processes:
            proc.terminate()
        return
    
    # Monitor all processes
    all_processes = server_processes + [client_proc]
    process_names = [f"Server-{i}" for i in range(3)] + ["Clients"]
    
    try:
        monitor_processes(all_processes, process_names)
    except KeyboardInterrupt:
        logging.info("Received interrupt, shutting down...")
        for proc in all_processes:
            proc.terminate()
    
    logging.info("Federation run complete")

if __name__ == "__main__":
    main()