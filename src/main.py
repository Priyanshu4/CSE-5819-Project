import torch
import argparse
from pathlib import Path
import pickle
import json
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
import hdbscan

# Absolute imports only work if ran from the root directory
cwd = Path.cwd()
if cwd != Path(__file__).parent.parent:
    raise RuntimeError(f"Please run this script from the project root.\n" + 
                       f"Use command python -m src.main in directory {Path(__file__).parent.parent.resolve()}")

from src.dataloader import DataLoader, YelpNycDataset
from src.config import DATASETS_CONFIG_PATH, get_results_path, get_logger
import src.utils as utils

from src.embedding.lightgcn import LightGCNTrainingConfig, LightGCNConfig, LightGCN
from src.embedding.loss import SimilarityLoss, BPRLoss
from src.embedding import training
from src.visualization.embvis import plot_embeddings, plot_embeddings_with_anomaly_scores
from src.visualization.trainvis import plot_loss_epochs


from src.clustering.hclust import HClust
from src.clustering.anomaly import AnomalyScorer
from src.clustering import split

from src.testing.dbscan import test_optics_dbscan_fraud_detection, log_dbscan_results
from src.testing.clust_anomaly import (test_clust_anomaly_fraud_detection, log_clust_anomaly_results, userwise_anomaly_scores)


def embedding_main(args, dataset, results_path, logger):
    """ Main code for the lightgcn training and embedding generation.
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
        loss = SimilarityLoss(device, dataset, n_pos=10, n_neg=15, fast_sampling=args.fast_simi)
        if loss.fast_sampling:
            logger.info(f"Adjusted n_pos {loss.n_pos}, n_neg {loss.n_neg}")
        train_lightgcn = training.train_lightgcn_simi_loss
    else:
        logger.error(f"Loss function {args.loss} is not supported.")
        raise ValueError(f"Loss function {args.loss} is not supported.")
            
    logger.info(f"Training LightGCN for {args.epochs} epochs on {args.dataset} dataset.")
    logger.info(f"Training with {loss.__class__.__name__} loss and {optimizer.__class__.__name__} optimizer.")
    logger.info(f"LightGCN configured to produce {lightgcn_config.latent_dim} dimensional embeddings.")

    losses = np.zeros(train_config.epochs)
    for epoch in range(train_config.epochs):
        losses[epoch] = train_lightgcn(dataset, lightgcn, loss, optimizer, epoch, logger)

    if args.plot:
        loss_plot_file = results_path / "loss.png"
        plot_loss_epochs(losses, loss_plot_file)
        logger.info(f"Saved a plot of loss over epochs to {loss_plot_file}")
        
    user_embs, item_embs = lightgcn()
    embeddings_save_file = results_path / "embeddings.pkl"
    pickle.dump(user_embs, open(embeddings_save_file, 'wb'))
    logger.info(f"Saved user embeddings to {embeddings_save_file}")
    
    return user_embs.detach().cpu().numpy()

def clustering_main(args, dataset, user_embs, results_path, logger):
    """ Main code for the clustering.
    """
    if args.clustering == "none":
        logger.info("Skipping clustering and fraud detection.")
        return

    n_users, n_features = user_embs.shape
    logger.info(f"Clustering {n_users} users with {n_features} features.")

    if args.clustering in ("hdbscan", "hclust"):

        use_metadata = args.metadata and (type(dataset) == YelpNycDataset) # Only the YelpNYC dataset has metadata avaiable
        logger.info(f"use_metadata: {use_metadata}")

        if args.clustering == "hdbscan":
            logger.info("Clustering with HDBSCAN for hieararchical with density and anomaly score based fraud detection.")

            with utils.timer(name="clustering"):
                min_cluster_size = 100
                hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                hdbscan_clusterer.fit(user_embs)

            with utils.timer(name="plotting"):
                if args.plot:
                    plt.figure(figsize=(10, 8))
                    hdbscan_clusterer.condensed_tree_.plot()
                    hdbscan_tree_fig_path = results_path / "hdbscan_tree.png"
                    plt.savefig(hdbscan_tree_fig_path)
                    plt.close()
                    logger.info(f"Saved condensed tree plot to {hdbscan_tree_fig_path}")

            with utils.timer(name="anomaly_scores"):
                anomaly_scorer = AnomalyScorer(dataset, enable_penalty=True, use_metadata=use_metadata, burstness_threshold=args.tau)
                groups, anomaly_scores = anomaly_scorer.hdbscan_tree_anomaly_scores(hdbscan_clusterer.condensed_tree_.to_pandas())
                clusters = [g.users for g in groups]

            time_info = utils.timer.formatted_tape_str(select_keys=["clustering", "plotting", "anomaly_scores"])
            logger.info(f"HDBSCAN Time: {time_info}")
            utils.timer.zero(select_keys=["clustering", "plotting", "anomaly_scores", "group_mapping"])

        elif args.clustering == "hclust":
            logger.info(f"Clustering with hierarchical clustering ({args.linkage} linkage) and anomaly scores for fraud detection.")
        
            max_group_size = 60000
            split_groups, group_indices = split.split_matrix_kmeans(user_embs, max_group_size=max_group_size)
            mappings = split.build_group_split_mappings(split_groups, group_indices)
            logger.info(f"Split {n_users} into {len(split_groups)} with max size {max_group_size}.")

            enable_penalty = (args.clustering == "hclust")   # Only enable penalty for hclust not hclust2
            anomaly_scorer = AnomalyScorer(dataset, enable_penalty=enable_penalty, use_metadata=use_metadata, burstness_threshold=args.tau)

            clusters = []
            anomaly_scores = []

            for i, group in enumerate(split_groups):

                logger.info(f"Clustering group {i} ({len(group)} users)")
                with utils.timer(name="linkages"):
                    hclust = HClust(group)
                    linkage = hclust.generate_linkage_matrix(method=args.linkage)

                logger.info(f"Computing anomaly scores for group {i}")
                with utils.timer(name="anomaly_scores"):
                    group_clusters, group_anomaly_scores = anomaly_scorer.hierarchical_anomaly_scores(linkage, mappings[i])
                    clusters.extend(g.users for g in group_clusters)
                    anomaly_scores.extend(group_anomaly_scores)

            time_info = utils.timer.formatted_tape_str(select_keys=["linkages", "anomaly_scores"])
            logger.info(f"Clustering Time: {time_info}")
            utils.timer.zero(select_keys=["linkages", "anomaly_scores"])

        max_anomaly_score = np.max(anomaly_scores)
        scale_factor = 1 if max_anomaly_score <= 1 else 1 / max_anomaly_score
        scaled_anomaly_scores = anomaly_scores * scale_factor

        logger.info(f"Anomaly scores scaled by {scale_factor}.")
        logger.info("Anomaly score statistics:")
        logger.info(f"\tmin={np.min(scaled_anomaly_scores)}")
        logger.info(f"\tmax={np.max(scaled_anomaly_scores)}")
        logger.info(f"\tmean={np.mean(scaled_anomaly_scores)}")
        logger.info(f"\tmedian={np.median(scaled_anomaly_scores)}")
        logger.info(f"\tstd={np.std(scaled_anomaly_scores)}")


        if args.plot:
            anomaly_score_plot_path = results_path / "anomaly_scores.png"
            user_anomaly_scores = userwise_anomaly_scores(clusters, anomaly_scores, n_users)
            plot_embeddings_with_anomaly_scores(user_embs, user_anomaly_scores, anomaly_score_plot_path)
            logger.info(f"Saved anomaly score plot to {anomaly_score_plot_path}")
            
        thresholds = list(np.linspace(0, 0.9, 9, endpoint=False)) + list(np.linspace(0.9, 0.99, 9, endpoint=False)) + list(np.linspace(0.99, 1, 11))     
        results, best = test_clust_anomaly_fraud_detection(clusters, scaled_anomaly_scores, thresholds, dataset.user_labels)
        log_clust_anomaly_results(thresholds, results, best, logger)

    elif args.clustering == "dbscan":
        logger.info("Clustering with DBSCAN for density based fraud detection.")
        epsilon_values = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
        min_samples_values = [5, 10, 20, 50]
        results = test_optics_dbscan_fraud_detection(user_embs, 0.05, epsilon_values, min_samples_values, dataset.user_labels, logger)
        log_dbscan_results(results, logger)
    

if __name__ == "__main__":

    dataloader = DataLoader(DATASETS_CONFIG_PATH)
    
    parser = argparse.ArgumentParser(description="Fake Review Group Detection with LightGCN embeddings and clustering.")

    # General arguments
    parser.add_argument("--name", type=str, default="", help="The experiment name for the results folder.")
    parser.add_argument("--dataset", type=str, default="yelpnyc", help=f"available datasets: {dataloader.dataset_names}")
    parser.add_argument("--seed", type=int, default=5819, help="random seed")
    parser.add_argument("--plot", action="store_true", help="save plots to results folder")
    parser.add_argument("--no_plot", action="store_false", dest="plot", help="save plots to results folder")
    parser.set_defaults(plot=True)

    # Arguments for embedding and LightGCN
    parser.add_argument("--embeddings", type=str, default="", help="The path to the embeddings file. If given, training is skipped and clustering is done directly.")
    parser.add_argument("--epochs", type=int, default=100, help="the epochs for training")
    parser.add_argument("--batch_size", type=int, default=2048, help="the batch size for training procedure")
    parser.add_argument("--dim", type=int, default=16, help="the embedding size of lightGCN")
    parser.add_argument("--layer", type=int, default=3, help="the layer num of lightGCN")
    parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")
    parser.add_argument("--decay", type=float, default=1e-4, help="the weight decay for l2 normalizaton")
    parser.add_argument("--dropout", action="store_true", help="enable dropout")
    parser.add_argument("--no_dropout", action="store_false", dest="dropout", help="disable dropout")
    parser.set_defaults(dropout=False)
    parser.add_argument("--keepprob", type=float, default=0.6, help="the dropout keep prob")
    parser.add_argument("--a_fold", type=int, default=100, help="the fold num used to split large adj matrix")
    parser.add_argument("--loss", type=str, default="simi", help="loss function, options: bpr, simi")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer, options: adam, sgd")
    parser.add_argument("--fast_simi", action="store_true", help="faster sampling for simi loss, use for very large & sparse datasets")
    parser.add_argument("--no_fast_simi", action="store_false", dest="fast_simi", help="disable fast sampling for simi loss")
    parser.set_defaults(fast_simi=True)

    # Arguments for clustering and anomaly scores
    parser.add_argument("--clustering", type=str, default="hclust", help="The clustering algorithm to use. Options: hclust, hdbscan, dbscan, none")
    parser.add_argument("--linkage", type=str, default="average", help="The linkage to use in hierarchical clustering. Options: single, average, ward")
    parser.add_argument("--no_metadata", action="store_false", dest="metadata", help="Do not use metadata in anomaly score computation.")
    parser.add_argument("--tau", type=float, default=30, help="The threshold for burstness for anomaly score computation in units of days.")
 
    args = parser.parse_args()

    if args.name:
        experiment_name = args.name + "_" + utils.current_timestamp()
    else:
        experiment_name = utils.current_timestamp()
        
    results_path = get_results_path(experiment_name)

    # Write a json file with the experiment parameters
    json.dump(vars(args), open(results_path / "params.json", 'w'), indent=4)

    # Get Logger
    logger = get_logger("Logger", results_path, args.name)
    sys.stderr = utils.StreamToLogger(logger, logging.ERROR)

    # Load Dataset
    logger.info(f"Loading dataset {args.dataset}.")
    dataset = dataloader.load_dataset(args.dataset)

    # Set seed
    utils.set_seed(args.seed)
    logger.info(f"SEED: {args.seed}")

    if args.embeddings:
        # If embeddings are already given, skip training.
        logger.info(f"Skipping training. Loading embeddings from {args.embeddings}")
        user_embs = dataloader.load_user_embeddings(args.embeddings)
    else:
        user_embs = embedding_main(args, dataset, results_path, logger)

    if args.plot:
        embeddings_plot_save_file = results_path / "embeddings.png"
        plot_embeddings(user_embs, dataset.user_labels, embeddings_plot_save_file)
        logger.info(f"Saved embeddings plot to {embeddings_plot_save_file}")

    clustering_main(args, dataset, user_embs, results_path, logger)

    




