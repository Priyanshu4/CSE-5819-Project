import torch
import numpy as np
from datetime import datetime

from src.embedding.sampling import set_sampling_seed

def current_timestamp():
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M")
    return timestamp

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

def flatten_list(list_of_lists: list) -> list:
    """
    Flatten a list of lists into a single list.
    
    Arguments:
        list_of_lists: A list of lists
        
    Returns:
        flat_list: A flattened list
    """
    flat_list = []
    return map(flat_list.extend, list_of_lists)


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        self.linebuf += buf
        while '\n' in self.linebuf:
            line, self.linebuf = self.linebuf.split('\n', 1)
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        if self.linebuf:
            self.logger.log(self.level, self.linebuf.rstrip())
            self.linebuf = ''

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

class IdentityMap:
    """
    A map that returns the key as the value.
    """

    def __init__(self):
        pass

    def __getitem__(self, key):
        return key

def print_table(headers: list, data: list, print_func = print):
    """
    Print a table with the first column left-justified and the remaining columns right-justified.

    Arguments:
        headers: List of headers for the columns.
        data: List of dictionaries, where each dictionary represents a row in the table.
        print_func: Optional function to use to print the table. Defaults to print.
    """

    # Find the maximum width of each column including the header
    widths = [len(header) for header in headers]
    for row in data:
        for i, header in enumerate(headers):
            widths[i] = max(widths[i], len(str(row.get(header, ''))))

    # Define a format string with the first column left-justified
    # and the rest right-justified based on the widths
    format_string = "{:<" + str(widths[0]) + "}"  # Left justify the first column
    for w in widths[1:]:
        format_string += " {:>" + str(w) + "}"  # Right justify the remaining columns

    # Print the header row
    print_func(format_string.format(*headers))

    # Print the separator
    print_func("-" * (sum(widths) + len(headers) - 1))

    # Print the data rows
    for row in data:
        print_func(format_string.format(*[row.get(header, '') for header in headers]))

