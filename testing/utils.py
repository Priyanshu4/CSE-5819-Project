import pickle
import torch
from pathlib import Path
import numpy as np

def load_user_embeddings(file: Path):
    """
    Loads user embeddings from a pickle file. The pickle file should contain a 2D numpy array or a torch tensor
    with shape (users, features), where each row represents the embedding of a user.

    Args:
    file_name: A string representing the file path of the pickle file.

    Returns:
    A numpy array of user embeddings.
    """
    with open(file, "rb") as file:
        try:
            loaded_object = pickle.load(file)
        except RuntimeError as e:
            tensor = torch.load(file, map_location=torch.device('cpu'))
        if isinstance(loaded_object, torch.Tensor):
            loaded_object = loaded_object.detach().numpy()
        if not isinstance(loaded_object, np.ndarray):
            raise ValueError("The pickle file should contain a 2D numpy array or a torch tensor.")
    return loaded_object


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

