from pathlib import Path
import logging
import sys
from .utils import current_timestamp

CONFIGS_PATH = Path(__file__).parent.parent / "configs"
DATASETS_CONFIG_PATH = CONFIGS_PATH / "datasets.json"
ALL_RESULTS_PATH = Path(__file__).parent.parent / "results"

LOGGING_LEVEL = logging.INFO

def get_logger(name: str, log_dir: Path, filename: str = "") -> logging.Logger:

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)

    if not log_dir.exists():
        log_dir.mkdir()
        
    # Create a file handler which logs messages
    if filename:
        log_file = log_dir / (filename + "_" + current_timestamp() + ".log")
    else:
        log_file = log_dir / (current_timestamp() + ".log")
    file_handler = logging.FileHandler(log_file)
    file_format = '%(asctime)s - [%(levelname)s] - [%(name)s] - %(message)s'
    file_handler.setFormatter(logging.Formatter(file_format))

    # Create a console handler which logs messages to stdout
    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(std_out_format))

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_results_path(name: str):
    """ Returns the path to the results of the experiment with the given name.
    """
    if not ALL_RESULTS_PATH.exists():
        ALL_RESULTS_PATH.mkdir()

    results_path = ALL_RESULTS_PATH / name

    if not results_path.exists():
        results_path.mkdir()

    return results_path.resolve()