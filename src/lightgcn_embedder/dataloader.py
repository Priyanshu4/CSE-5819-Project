from pathlib import Path
import pickle
import json
import scipy.sparse
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import torch

class DataLoader:

    def __init__(self, datasets_config_file: Path):
        
        if not datasets_config_file.exists():
            raise FileNotFoundError(f"Create a Datasets Path Config file at {datasets_config_file}!")

        self.datasets_config_file = datasets_config_file
        self.datasets_config = json.load(open(datasets_config_file))

    @property
    def dataset_names(self):
        return self.datasets_config.keys()
    
    def parse_path(self, path: str):
        """ Parse a path string from the datasets config file into a Path object
            If the path is relative, the path is assumed to be relative to the datasets config file
        """
        if Path(path).is_absolute():
            return Path(path)
        else:
            return (Path(self.datasets_config_file.parent) / path).resolve()

    def load_dataset(self, dataset_name: str):
        if dataset_name not in self.dataset_names:
            raise ValueError(f"{dataset_name} is not a valid dataset.")
        
        dataset_config = self.datasets_config[dataset_name]

        if "dataset_type" not in dataset_config.keys():
            raise ValueError(f"dataset_type not specified for {dataset_name} dataset.")
        
        try:
            if dataset_config["dataset_type"] == "pickle":
                dataset = PickleDataset(
                    u2i_pkl_path=self.parse_path(dataset_config["filepaths"]["graph_u2i"]),
                    user_labels_pkl_path=self.parse_path(dataset_config["filepaths"]["labels"]),
                )
            elif dataset_config["dataset_type"] == "yelpnyc":
                dataset = YelpNycDataset(
                    metadata_csv_path=self.parse_path(dataset_config["filepaths"]["metadata"]),
                )
            else:
                raise ValueError(f"{dataset_config['dataset_type']} is not a valid dataset type.")
        except KeyError as e:
            raise KeyError(f"Missing key in {dataset_name} dataset config: {e}")

        return dataset
    
    def load_user_embeddings(self, file: Path):
        """
        Loads user embeddings from a pickle file. The pickle file should contain a 2D numpy array or a torch tensor
        with shape (users, features), where each row represents the embedding of a user.

        Args:
        file: A pathlike representing the file path of the pickle file.

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
    
class GraphUser2ItemDataset(BasicDataset):
    """ Superclass for easily creating datasets with the user to item matrix.
        The user to item sparse matrix and labels array are the only required input.
        Other graphs are generated automatically.
    """

    def __init__(self, graph_u2i: scipy.sparse.csr_matrix, user_labels: np.array):
        self._graph_u2i = graph_u2i
        self._graph_u2u = self._graph_u2i @ self._graph_u2i.T
        self._labels = user_labels
        self._n_users, self._m_items = self._graph_u2i.shape
        if len(self._labels) != self._n_users:
            raise ValueError(
                "Number of labels does not match number of users from graph!"
            )
        self._graph_adj_mat = self.build_adj_mat(self._graph_u2i)
        
    def build_adj_mat(self, graph_u2i: scipy.sparse.csr_matrix):
        """
        Build full adjancency matrix of size (n_users + m_items, n_users + m_items) 
        Users have no connections to each other and items have no connections to each other
        """
        users, items = graph_u2i.shape
        zero_matrix_user = scipy.sparse.csr_matrix((users, users))
        zero_matrix_item = scipy.sparse.csr_matrix((items, items))        
        graph_adj_mat = scipy.sparse.vstack([
            scipy.sparse.hstack([zero_matrix_user, graph_u2i]),
            scipy.sparse.hstack([graph_u2i.T, zero_matrix_item])
        ]).tocsr()
        return graph_adj_mat

    @property
    def n_users(self):
        return self._n_users

    @property
    def m_items(self):
        return self._m_items

    @property
    def n_interactions(self):
        return self._graph_u2i.nnz
    
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
        
class PickleDataset(BasicDataset):
    """ For loading dataset from pickled u2i csr matrix and labels array.
    """
    def __init__(self, u2i_pkl_path: Path, user_labels_pkl_path: Path):
        graph_u2i = pickle.load(open(u2i_pkl_path, "rb"))
        labels = pickle.load(open(user_labels_pkl_path, "rb"))
        super().__init__(graph_u2i, labels)
        
class YelpNycDataset(GraphUser2ItemDataset):
    """ Dataset for YelpNYC
        Contains additional metadata including star ratings, timestamps, user ids and product ids
    """
    # Define metadata column names
    METADATA_REVIEWER_ID = "Reviewer_id"
    METADATA_USER_ID = "Reviewer_id" # Alias for reviewer id
    METADATA_PRODUCT_ID = "Product_id"
    METADATA_ITEM_ID = "Product_id" # Alias for product id
    METADATA_STAR_RATING = "Rating"
    METADATA_LABEL = "Label"
    METADATA_DATE = "Date" 

    # Define metadata label values
    METADATA_FRAUD_LABEL = -1
    METADATA_NON_FRAUD_LABEL = 1

    # Define date format
    METADATA_DATE_FORMAT = "%m/%d/%Y"
    
    def __init__(self, metadata_csv_path: Path):
        self.metadata_df = pd.read_csv(metadata_csv_path)
        self.metadata_df[self.METADATA_DATE] = pd.to_datetime(
            self.metadata_df[self.METADATA_DATE], format=self.METADATA_DATE_FORMAT)
        self._build_graphs()
        super().__init__(self._graph_u2i, self._labels)
    
    def _build_graphs(self):
        df = self.metadata_df
        unique_reviewers = df[self.METADATA_REVIEWER_ID].unique()
        unique_reviewers.sort()
        unique_products = df[self.METADATA_PRODUCT_ID].unique()
        unique_products.sort()
        num_reviews = len(df)
        num_reviewers = len(unique_reviewers)
        num_products = len(unique_products)

        # Create dictionaries to map reviewers and products to matrix indices
        self.reviewer_to_index = {reviewer: index for index, reviewer in enumerate(unique_reviewers)}
        self.product_to_index = {product: index for index, product in enumerate(unique_products)}
        self.item_to_index = self.product_to_index # Item is alias for product

        # Convert the 'Reviewer_id' and 'Product_id' columns to NumPy arrays
        reviewer_ids = df[self.METADATA_REVIEWER_ID].apply(lambda x: self.reviewer_to_index[x]).values
        product_ids = df[self.METADATA_PRODUCT_ID].apply(lambda x: self.product_to_index[x]).values
        ratings = df[self.METADATA_STAR_RATING].values

        # Create the user-to-item matrix
        ones = np.ones(num_reviews)
        self._graph_u2i = csr_matrix((ones, (reviewer_ids, product_ids)), shape=(num_reviewers, num_products))
        
        # Create the user-to-item matrix with star ratings
        ratings = df[self.METADATA_STAR_RATING].values
        self._rated_graph_u2i = csr_matrix((ratings, (reviewer_ids, product_ids)), shape=(num_reviewers, num_products))
        self._rated_adj_mat = self.build_adj_mat(self._rated_graph_u2i)

        reviewer_labels = df.groupby('Reviewer_id')['Label'].apply(
            lambda x: 1 if self.METADATA_FRAUD_LABEL in x.values else 0)
        reviewer_labels.index = reviewer_labels.index.map(self.reviewer_to_index)
        self._labels = reviewer_labels.values

    @property
    def rated_graph_u2i(self):
        """User to item graph as scipy sparse matrix of shape (n_users, m_items)
           The matrix contains star ratings instead of 1s and 0s. 
           A 0 indicates no interaction.
        """
        return self._rated_graph_u2i
    
    @property
    def rated_adj_mat(self):
        """Adjacency matrix of user to item graph as scipy sparse matrix.
           The matrix has shape (n_users + m_items, n_users + m_items).
           The matrix contains star ratings instead of 1s and 0s.
           A 0 indicates no interaction.
        """
        return self._rated_adj_mat

    






