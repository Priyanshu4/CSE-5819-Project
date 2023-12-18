# Fake-Review-Farm-Detection

## About 
As part of a machine learning course at the University of Connecticut (CSE 5819), we proposed, implemented and tested a novel model for fake review detection from user and item structure data. Our model, like many other fake review group detection models, finds dense groups of reviewers with highly similar behavior. If many reviewers have highly similar review patterns, they are likely a paid group of fake reviewers.

The input to our model is a bipartite graph of users and items, where an edge between a user and an item indicates that the user has reviewed the item. We use a light graph convolutional neural network to generate embeddings of the user nodes. We then use hierarchical density based clustering to generate candidate fraud clusters. These clusters are scored on metrics which also incorporate review metadata. If a candidate group has a score higher than a threshold, it is fraudulent. Unfortunately, our approach did not have strong performance. Please see the file `presentation.pdf` for details on our algorithm and our results. The entire project, including conceptualization, implementation, testing and presentation was completed in only 3 months during a college semester.

## Environment Setup
To use our repository, we recommend setting up a conda environment. You can install all the necessary packages with the following conda commands.
Replace the lines for PyTorch installation with the correct line for your operating system and CUDA version if applicable. Note that a GPU is not required. Even with a CPU only PyTorch our model is still relatively fast.

```bash
conda create -n fake-review-farm-detection python=3.10
conda activate fake-review-farm-detection 
conda install -c conda-forge hdbscan

# Replace the following lines to install PyTorch with your desired CUDA version and OS
# See https://pytorch.org/get-started/locally/
# conda install pytorch torchvision torchaudio cpuonly -c pytorch                     # For Linux with CPU only
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # For Linux with CUDA 12.1

conda install scipy
conda install scikit-learn
conda install seaborn
conda install bitarray
```

## Usage
To run our code, activate the conda environment. You can run `python -m src.main -h` from the root directory to see the full list of command line arguments and explanations for each. As an example to run on the `synthetic_10k` dataset with 200 epochs of LightGCN, 8 dimensional latent space, and HDBSCAN clustering, we can use the following command:

`python -m src.main --name synthetic10k_test --dataset synthetic_10k --epochs 200 --dim 8 --clustering hdbscan`

This command should work straight out of the box after cloning our repository and setting up the environment. In this command, the argument `--name synthetic10k_test` specifies the experiment name for the results folder. If a folder called `results` does not exist in the project root it will be automatically created. Within that folder, a subdirectory called `synthetic10k_test_{timestamp}` will be created. This folder will contain the log file in addition to various plots such as the embeddings plot, the training loss over epochs, the predicted scores on the embeddings and more.

## Dataset Configuration
You can easily use custom datasets with our code by adding them to the data folder. You will then need to add the dataset to the config file at `config/datasets.json`. If the dataset is in the format of a pickled user to item sparse matrix and a pickled numpy array for labels, it can easily be added as a pickle dataset (see synthetic_10k) dataset. In this case, no extra code is required. If the dataset is in a custom format like YelpNYC, you will need to extend one of the dataset classes in dataloader.py and create your own dataset type.

### YelpNYC Dataset
The YelpNYC dataset is the main dataset we use to test our algorithm. Our repository already contains the pickled version of the dataset without metadata, which can be used with `--dataset yelpnyc_pickle`. To use the actual YelpNYC dataset with metadata, unzip the zip in the YelpNYC folder. Make sure that the path of the metadata file matches the path expected by the `datasets.json` file.











