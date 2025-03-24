import logging
from datetime import datetime
from pathlib import Path

def setup_logging():
    """Configure logging settings for the entire project."""
    # Ensure logs directory exists
    logs_dir = Path("../logs")
    logs_dir.mkdir(exist_ok=True)

    # Define log filename with timestamp to avoid overwriting
    log_filename = logs_dir / f"client_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging initialized. Logs will be saved to: {log_filename}")
