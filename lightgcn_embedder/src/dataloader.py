from pathlib import Path
import pickle
import json
import scipy.sparse

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
    def graph_adj_mat(self):
        """Adjacency matrix of user to item graph as scipy sparse matrix.
           The matrix has shape (n_users + m_items, n_users + m_items) 
        """
        raise NotImplementedError
    
    @property
    def graph_u2i(self):
        """User to item graph as scipy sparse matrix of shape (n_users, m_items)"""
        raise NotImplementedError

    @property
    def graph_u2u(self):
        """User to user relationships as scipy sparse matrix of shape (n_users, n_users)
           This is not the adjacency matrix of users, instead this is u2i @ u2i.T
        """
        raise NotImplementedError

    @property
    def user_labels(self):
        """Node labels as numpy array
        1 indicates fradulent user
        0 indicates non-fradulent user
        """
        raise NotImplementedError

    def get_adj_mat_split(self, folds):
        """ Returns the adjacency matrix split into a list of folds.
            The list has length equal to the number of folds.
            Each fold has shape (n+m)/n_folds x (n+m)
        """
        adj_folds = []
        fold_len = (self.n_users + self.m_items) // folds
        for i_fold in range(folds):
            start = i_fold*fold_len
            if i_fold == folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            adj_folds.append(self.graph_adj_mat[start:end].copy())
        return adj_folds

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
        
        # Build full adjancency matrix of size (n_users + m_items, n_users + m_items) 
        # Users have no connections to each other and items have no connections to each other
        zero_matrix_user = scipy.sparse.csr_matrix((self._n_users, self._n_users))
        zero_matrix_item = scipy.sparse.csr_matrix((self._m_items, self._m_items))        
        self._graph_adj_mat = scipy.sparse.vstack([
            scipy.sparse.hstack([zero_matrix_user, self._graph_u2i]),
            scipy.sparse.hstack([self._graph_u2i.T, zero_matrix_item])
        ]).tocsr()

    @property
    def n_users(self):
        return self._n_users

    @property
    def m_items(self):
        return self._m_items

    @property
    def graph_adj_mat(self):
        """Adjacency matrix of user to item graph as scipy sparse matrix.
           The matrix has shape (n_users + m_items, n_users + m_items) 
        """
        return self._graph_adj_mat
    
    @property
    def graph_u2i(self):
        """User to item graph as scipy sparse matrix of shape (n_users, m_items)"""
        return self._graph_u2i

    @property
    def graph_u2u(self):
        """User to user relationships as scipy sparse matrix of shape (n_users, n_users)
           This is not the adjacency matrix of users, instead this is u2i @ u2i.T
        """
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



