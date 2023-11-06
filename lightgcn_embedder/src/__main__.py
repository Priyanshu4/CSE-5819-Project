import utils
import torch
import numpy as np
import time
import dataloader
from os.path import join
import parse_args
import multiprocessing
from lightgcn import LightGCNTrainingConfig, LightGCNConfig, LightGCN

if __name__ == "__main__":
    args = parse_args.parse_args()

    dataset_paths = dataloader.dataset_paths()
    if args.dataset not in dataset_paths.keys():
        print("Invalid dataset selected.")
        print(f"Options are {dataset_paths.keys()}")
        exit(-1)

    dataset = dataloader.PickleDataset(
        u2i_pkl_path=dataset_paths[args.dataset]["graph_u2i"],
        user_labels_pkl_path=dataset_paths[args.dataset]["labels"],
    )

    GPU = torch.cuda.is_available()
    device = torch.device("cuda" if GPU else "cpu")
    CORES = multiprocessing.cpu_count() // 2

    # ==============================
    utils.set_seed(args.seed)
    print(">>SEED:", args.seed)
    # ==============================

    # Build Config
    train_config = LightGCNTrainingConfig(
        
    )

    lightgcn_config = LightGCNConfig(
        latent_dim=args.recdim,
        n_layers=args.layer,
        keep_prob=args.keepprob,
        A_split=args.a_fold,
        dropout=bool(args.dropout),
    )

    lightgcn = LightGCN(lightgcn_config, dataset)
    lightgcn = lightgcn.to(device)
    bpr = utils.BPRLoss(Recmodel, world.config)

    Neg_k = 1


    for epoch in range(args.epochs):
        start = time.time()
        output_information = Procedure.BPR_train_original(
                dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
            )
        print(f"EPOCH[{epoch+1}/{args.epochs}] {output_information}")
