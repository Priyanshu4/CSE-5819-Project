import torch
import argparse
from pathlib import Path
import pickle

from src.dataloader import DataLoader
import src.config as config
import src.utils as utils
from .similarity import GraphSimilarity
from .lightgcn import LightGCNTrainingConfig, LightGCNConfig, LightGCN
from .loss import SimilarityLoss, BPRLoss
from . import training

CONFIGS_PATH = config.CONFIGS_DIRECTORY_PATH
DATASET_CONFIG = config.DATASETS_CONFIG_PATH
LOGS_PATH = Path(__file__).parent.parent / "results" / "logs"
EMBEDDINGS_PATH = Path(__file__).parent.parent / "results" / "embeddings"

if __name__ == "__main__":

    dataloader = DataLoader(DATASET_CONFIG)
    
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

    GPU = torch.cuda.is_available()
    device = torch.device("cuda" if GPU else "cpu")

    if GPU: 
        logger.info(f"{torch.cuda.get_device_name(torch.cuda.current_device())} will be used for training.")
    else:
        logger.info(f"No GPU available. CPU will be used for training.")

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
            loss = SimilarityLoss(device, dataset, GraphSimilarity(dataset.graph_u2u), n_pos=10, n_neg=11, fast_sampling=True)
        else:
            loss = SimilarityLoss(device, dataset, GraphSimilarity(dataset.graph_u2u), n_pos=10, n_neg=10, fast_sampling=False)
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
    if args.name:
        embeddings_save_file = EMBEDDINGS_PATH / f'embs_{args.name}_{utils.current_timestamp()}.pkl'
    else:
        embeddings_save_file = EMBEDDINGS_PATH / f'embs_{utils.current_timestamp()}.pkl'

    pickle.dump(user_embs.detach().cpu().numpy(), open(embeddings_save_file, 'wb'))
    logger.info(f"Saved user embeddings to {embeddings_save_file}")