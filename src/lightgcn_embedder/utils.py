import torch
import numpy as np
import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime
from sampling import set_sampling_seed


def current_timestamp():
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M")
    return timestamp

def configure_logger(name: str, log_dir: Path, filename: str = "", log_level: str = "info") -> logging.Logger:
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(log_levels.get(log_level, logging.INFO))

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

def set_seed(seed: int):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    set_sampling_seed(seed)


def sparse_matrix_to_tensor(X) -> torch.sparse_coo_tensor:
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
    

class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def format_time_dhms(seconds):
        """
        Format time in seconds to days, hours, minutes, and seconds
        For example, 10000 seconds is 2h 46m 40s
        """
        # Calculate days, hours, minutes and seconds
        days, remaining_seconds = divmod(seconds, 86400)
        hours, remaining_seconds = divmod(remaining_seconds, 3600)
        minutes, remaining_seconds = divmod(remaining_seconds, 60)

        time_str = f"{remaining_seconds:05.2f}s"
        if seconds >= 60:
            time_str = f"{int(minutes):02d}m " + time_str
        if seconds >= 3600:
            time_str = f"{int(hours):02d}h " + time_str
        if seconds >= 86400:
            time_str = f"{int(days):01d}d " + time_str

        return time_str
    
    @staticmethod
    def formatted_tape_str(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{timer.format_time_dhms(value)}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{timer.format_time_dhms(value)}|"
        return hint
    
    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint


    
    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)