from pathlib import Path
import pickle
import json


class DataLoader:

    def __init__(self, datasets_config_file: Path):
        
        if not datasets_config_file.exists():
            raise FileNotFoundError(f"Create a Datasets Path Config file at {datasets_config_file}!")

        self.datasets_config = json.load(open(datasets_config_file))

    @property
    def dataset_names(self):
        return self.datasets_config.keys()

    def load_dataset(self, dataset_name: str):
        if dataset_name not in self.dataset_names:
            raise ValueError(f"{dataset_name} is not a valid dataset.")
        
        dataset_config = self.datasets_config[dataset_name]

        if "dataset_type" not in dataset_config.keys():
            raise ValueError(f"dataset_type not specified for {dataset_name} dataset.")
        
        try:
            if dataset_config["dataset_type"] == "pickle":

                dataset = PickleDataset(
                    u2i_pkl_path=dataset_config["filepaths"]["graph_u2i"],
                    user_labels_pkl_path=dataset_config["filepaths"]["labels"],
                )
            else:
                raise ValueError(f"{dataset_config['dataset_type']} is not a valid dataset type.")
        except KeyError as e:
            raise KeyError(f"Missing key in {dataset_name} dataset config: {e}")

        return dataset
    
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
        self._graph_u2i = pickle.load(open(u2i_pkl_path, "rb"))
        self._graph_u2u = self._graph_u2i @ self._graph_u2i.T
        self._labels = pickle.load(open(user_labels_pkl_path, "rb"))
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