from pathlib import Path
import pickle
import json

CONFIGS_PATH = Path("configs")
DATASET_PATHS_JSON = CONFIGS_PATH / "dataset_paths.json"


def dataset_paths():
    if not DATASET_PATHS_JSON.exists():
        raise FileNotFoundError(f"Create a Datasets Path Config file at {DATASET_PATHS_JSON}!")

    with open(DATASET_PATHS_JSON, "r") as f:
        return json.load(f)

class BasicDataset:
    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def graph_u2i(self):
        """User to item graph as scipy sparse matrix"""
        raise NotImplementedError

    @property
    def graph_u2u(self):
        """User to user graph as scipy sparse matrix"""
        raise NotImplementedError

    @property
    def user_labels(self):
        """Node labels as numpy array
        1 indicates fradulent user
        0 indicates non-fradulent user
        """
        raise NotImplementedError


class PickleDataset(BasicDataset):
    def __init__(self, u2i_pkl_path: Path, user_labels_pkl_path: Path):
        self._graph_u2i = pickle.load(open(u2i_pkl_path))
        self._graph_u2u = self._graph_u2i @ self._graph_u2i
        self._labels = pickle.load(open(user_labels_pkl_path))
        self._n_users, self._m_items = self._graph_u2i.shape
        if len(self._labels) != self._n_users:
            raise ValueError(
                "Number of labels does not match number of users from graph!"
            )

    @property
    def n_users(self):
        return self._n_users

    @property
    def m_items(self):
        return self._m_items

    @property
    def graph_u2i(self):
        """User to item graph as scipy sparse matrix"""
        return self._graph_u2i

    @property
    def graph_u2u(self):
        """User to user graph as scipy sparse matrix"""
        return self._graph_u2u

    @property
    def user_labels(self):
        """Node labels as numpy array
        1 indicates fradulent user
        0 indicates non-fradulent user
        """
        return self._labels


class YelpDataset(PickleDataset):
    """ A pickle dataset that also has Yelp review metadata
    """
    pass