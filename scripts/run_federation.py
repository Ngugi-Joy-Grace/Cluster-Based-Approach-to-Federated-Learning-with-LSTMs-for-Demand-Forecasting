import subprocess
import threading
import time
import logging
import os
from configs.logging_config import get_run_timestamp_dir


def stream_process_output(proc: subprocess.Popen, name: str):
    """Continuously read lines from proc.stdout and log them."""
    if proc.stdout is None:
        return
    for line in proc.stdout:
        logging.info(f"[{name}] {line.rstrip()}")


def start_servers() -> list[subprocess.Popen]:
    """Start all federation servers and return their processes."""
    server_processes = []

    for cluster_id in range(3):  # For clusters 0, 1, and 2
        cmd = ["python", f"server_cluster_{cluster_id}.py"]
        try:
            # Pass the environment variable to the subprocess
            env = os.environ.copy()
            # Set up pipes but do not block on them
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # combine stderr into stdout
                universal_newlines=True,
                env=env
            )
            server_processes.append(proc)
            logging.info(f"Started server for cluster {cluster_id}")
            time.sleep(2)  # small delay for server init
        except subprocess.SubprocessError as e:
            logging.error(f"Failed to start server {cluster_id}: {e}")

    return server_processes

def start_clients() -> subprocess.Popen:
    """Start the client coordination process."""
    try:
        # Pass the environment variable to the subprocess
        env = os.environ.copy()
        cmd = ["python", "run_all_clients.py"] # Run the client coordinator
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        logging.info("Started client coordinator")
        return proc
    except subprocess.SubprocessError as e:
        logging.error(f"Failed to start clients: {e}")
        return None


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

    # Start reading server logs in background threads
    threads = []
    for i, proc in enumerate(server_processes):
        t = threading.Thread(target=stream_process_output, args=(proc, f"Server-{i}"), daemon=True)
        t.start()
        threads.append(t)

        # Start clients
        client_proc = start_clients()
        if not client_proc:
            logging.error("Failed to start clients")
            for p in server_processes:
                if p.poll() is None:
                    p.terminate()
            return

        # Read client coordinator logs in background thread
        t_client = threading.Thread(target=stream_process_output, args=(client_proc, "Clients"), daemon=True)
        t_client.start()
        threads.append(t_client)


    try:
        while True:
            time.sleep(5)  # sleep some seconds between checks
            # Check if all servers have finished
            still_running = [p for p in server_processes if p.poll() is None]
            if not still_running:
                logging.info("All servers have ended, stopping main script loop.")
                break

            logging.info(f"{len(still_running)} servers still running...")
    except KeyboardInterrupt:
        logging.info("Received interrupt, terminating servers...")
        for p in server_processes:
            if p.poll() is None:
                p.terminate()

    logging.info("Federation run complete")

if __name__ == "__main__":
    main()