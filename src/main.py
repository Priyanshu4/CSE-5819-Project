import torch
import argparse
from pathlib import Path
import pickle
import os
import multiprocessing
import numpy as np

from src.dataloader import DataLoader, YelpNycDataset
from src.config import DATASETS_CONFIG_PATH, LOGS_PATH, EMBEDDINGS_PATH
import src.utils as utils

from src.embedding.lightgcn import LightGCNTrainingConfig, LightGCNConfig, LightGCN
from src.embedding.loss import SimilarityLoss, BPRLoss
from src.embedding import training

from src.clustering.hclust import HClust
from src.clustering.anomaly import AnomalyScore
import src.clustering.split as split

def embedding_main(args, dataset, logger):
    """ Main code for the embedding generation.
    """
    GPU = torch.cuda.is_available()
    device = torch.device("cuda" if GPU else "cpu")

    if GPU: 
        logger.info(f"{torch.cuda.get_device_name(torch.cuda.current_device())} will be used for training.")
    else:
        logger.info(f"No GPU available. CPU will be used for training.")

    # Set configurations
    train_config = LightGCNTrainingConfig(
        epochs = args.epochs,
        batch_size = args.batch_size,
        learning_rate = args.lr,
        dropout = args.dropout,
        weight_decay = args.decay
    )

    lightgcn_config = LightGCNConfig(
        latent_dim = args.dim,
        n_layers = args.layer,
        keep_prob = args.keepprob,
        A_split = args.a_fold,
        device = device,
        train_config = train_config
    )

    lightgcn = LightGCN(lightgcn_config, dataset, logger)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(lightgcn.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(lightgcn.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    else:
        logger.error(f"Optimizer {args.optimizer} is not supported.")
        raise ValueError(f"Optimizer {args.optimizer} is not supported.")
    
    if args.loss == "bpr":
        loss = BPRLoss(device, dataset, weight_decay=train_config.weight_decay)
        train_lightgcn = training.train_lightgcn_bpr_loss
    elif args.loss == "simi":
        if args.fast_simi:
            # Fast sampling means that some negative samples will be positive samples.
            # Therefore, we increase the number of negative samples to compensate.
            loss = SimilarityLoss(device, dataset, n_pos=10, n_neg=11, fast_sampling=True)
        else:
            loss = SimilarityLoss(device, dataset, n_pos=10, n_neg=10, fast_sampling=False)
        train_lightgcn = training.train_lightgcn_simi_loss
    else:
        logger.error(f"Loss function {args.loss} is not supported.")
        raise ValueError(f"Loss function {args.loss} is not supported.")
            
    logger.info(f"Training LightGCN for {args.epochs} epochs on {args.dataset} dataset.")
    logger.info(f"Training with {loss.__class__.__name__} loss and {optimizer.__class__.__name__} optimizer.")
    logger.info(f"LightGCN configured to produce {lightgcn_config.latent_dim} dimensional embeddings.")

    for epoch in range(train_config.epochs):
        train_lightgcn(dataset, lightgcn, loss, optimizer, epoch, logger)

    user_embs, item_embs = lightgcn()

    return user_embs.detach().cpu().numpy()

def clustering_main(args, dataset, user_embs, logger):
    """ Main code for the clustering.
    """
    n_users, n_features = user_embs.shape
    logger.info(f"Clustering {n_users} users with {n_features} features.")

    if n_users > 60000:
        logger.warning(f"Splitting {n_users} into groups with max size 60000.")
        groups, group_indices = split.split_matrix_random(user_embs, approx_group_sizes=60000)
    else:
        groups, group_indices = split.split_matrix_random(user_embs, num_groups=1)

    def find_leaves_wrapper(args):
        clusterer,row = args
        return clusterer._find_leaves_iterative(row)

    all_clusters = []
    num_cores = os.cpu_count()
    for i, group in enumerate(groups):
        hclust = HClust(group)
        with utils.timer(name="linkages"):
            linkage = hclust.generate_linkage_matrix()

        with utils.timer(name="leaves"):
            with multiprocessing.Pool(num_cores) as p:
                # Find the leaves under every branch of the hierarchical clustering
                leaves = p.map(find_leaves_wrapper, [(hclust, row) for row in linkage])
                group_user_indices = np.array(group_indices[i])
                mapped_leaves = [group_user_indices[group] for group in leaves]
                all_clusters.extend(mapped_leaves)
   
    logger.info("Hclust Times:", utils.timer.formatted_tape_str(["linkages", "leaves"]))

    # Generate anomaly scores
    use_metadata = (type(dataset) == YelpNycDataset)
    anomaly_scorer = AnomalyScore(all_clusters, dataset, use_metadata=use_metadata, burstness_threshold=args.tau)
    anomaly_scores = anomaly_scorer.generate_anomaly_scores()

    # TODO: Make a testing script to test different anomaly score thresholds and compute metrics



if __name__ == "__main__":

    dataloader = DataLoader(DATASETS_CONFIG_PATH)
    
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument("--batch_size", type=int, default=2048, help="the batch size for training procedure")
    parser.add_argument("--dim", type=int, default=64, help="the embedding size of lightGCN")
    parser.add_argument("--layer", type=int, default=3, help="the layer num of lightGCN")
    parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")
    parser.add_argument("--decay", type=float, default=1e-4, help="the weight decay for l2 normalizaton")
    parser.add_argument("--dropout", type=int, default=0, help="using the dropout or not")
    parser.add_argument("--keepprob", type=float, default=0.6, help="the dropout keep prob")
    parser.add_argument("--a_fold", type=int, default=100, help="the fold num used to split large adj matrix")
    parser.add_argument("--dataset", type=str, default="yelpnyc", help=f"available datasets: {dataloader.dataset_names}")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5819, help="random seed")
    parser.add_argument("--loss", type=str, default="simi", help="loss function, options: bpr, simi")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer, options: adam, sgd")
    parser.add_argument("--fast_simi", action="store_true", help="faster sampling for simi loss, use for very large & sparse datasets")
    parser.add_argument("--name", type=str, default="", help="The name to add to the embs file and log file names.")
    args = parser.parse_args()

    logger = utils.configure_logger("Logger", LOGS_PATH, args.name, "info")
    logger.info(f"Loading dataset {args.dataset}.")
    dataset = dataloader.load_dataset(args.dataset)

    utils.set_seed(args.seed)
    logger.info(f"SEED: {args.seed}")

    user_embs = embedding_main(args, dataset, logger)

    if args.name:
        embeddings_save_file = EMBEDDINGS_PATH / f'embs_{args.name}_{utils.current_timestamp()}.pkl'
    else:
        embeddings_save_file = EMBEDDINGS_PATH / f'embs_{utils.current_timestamp()}.pkl'

    pickle.dump(user_embs, open(embeddings_save_file, 'wb'))
    logger.info(f"Saved user embeddings to {embeddings_save_file}")




