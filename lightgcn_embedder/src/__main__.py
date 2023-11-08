import utils
import torch
from dataloader import DataLoader
import multiprocessing
from similarity import GraphSimilarity
import argparse
from lightgcn import LightGCNTrainingConfig, LightGCNConfig, LightGCN
from loss import SimilarityLoss, BPRLoss
import training
from pathlib import Path

CONFIGS_PATH = Path(__file__).parent.parent / "configs"
DATASET_CONFIG = CONFIGS_PATH / "datasets.json"
LOGS_PATH = Path(__file__).parent.parent / "results" / "logs"

if __name__ == "__main__":

    dataloader = DataLoader(DATASET_CONFIG)
    
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument("--batch_size", type=int, default=2048, help="the batch size for training procedure")
    parser.add_argument("--recdim", type=int, default=64, help="the embedding size of lightGCN")
    parser.add_argument("--layer", type=int, default=3, help="the layer num of lightGCN")
    parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")
    parser.add_argument("--decay", type=float, default=1e-4, help="the weight decay for l2 normalizaton")
    parser.add_argument("--dropout", type=int, default=0, help="using the dropout or not")
    parser.add_argument("--keepprob", type=float, default=0.6, help="the dropout keep prob")
    parser.add_argument("--a_fold", type=int, default=100, help="the fold num used to split large adj matrix")
    parser.add_argument("--testbatch", type=int, default=100, help="the batch size of users for testing")
    parser.add_argument("--dataset", type=str, default="yelpnyc", help=f"available datasets: {dataloader.dataset_names}")
    parser.add_argument("--path", type=str, default="./checkpoints", help="path to save weights")
    parser.add_argument("--topks", nargs="?", default="[20]", help="@k test list")
    parser.add_argument("--comment", type=str, default="lgn")
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--multicore", type=int, default=0, help="whether we use multiprocessing or not in test")
    parser.add_argument("--pretrain", type=int, default=0, help="whether we use pretrained weight or not")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    args = parser.parse_args()

    logger = utils.configure_logger("Logger", LOGS_PATH, "info")
    dataset = dataloader.load_dataset(args.dataset)

    GPU = torch.cuda.is_available()
    device = torch.device("cuda" if GPU else "cpu")
    CORES = multiprocessing.cpu_count() // 2

    utils.set_seed(args.seed)
    logger.info(f"SEED: {args.seed}")

    # Set configurations
    train_config = LightGCNTrainingConfig(
        epochs = args.epochs,
        batch_size = args.batch_size,
        learning_rate = args.lr,
        dropout = args.dropout,
        weight_decay = args.decay
    )

    lightgcn_config = LightGCNConfig(
        latent_dim = args.recdim,
        n_layers = args.layer,
        keep_prob = args.keepprob,
        A_split = args.a_fold,
        device = device,
        train_config = train_config
    )

    lightgcn = LightGCN(lightgcn_config, dataset, logger)
    optimizer = torch.optim.Adam(lightgcn.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    loss = SimilarityLoss(dataset, GraphSimilarity(dataset.graph_u2u), n_pos=10, n_neg=10, fast_sampling=False)

    train_lightgcn = None
    if type(loss) == SimilarityLoss:
        train_lightgcn = training.train_lightgcn_simi_loss
    elif type(loss) == BPRLoss:
        train_lightgcn = training.train_lightgcn_bpr_loss
    else:
        raise TypeError(f"Loss of type {type(loss)} is not supported.")
            
    logger.info(f"Training LightGCN on {args.dataset} with {loss.__class__.__name__} loss function and {optimizer.__class__.__name__} optimizer.")

    for epoch in range(train_config.epochs):
        train_lightgcn(dataset, lightgcn, loss, optimizer, epoch, logger)