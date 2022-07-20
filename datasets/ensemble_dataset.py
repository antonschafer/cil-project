import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_base_datasets, load_wandb_file


class EnsembleDataset(Dataset):
    def __init__(self, runs, split, save_dir):
        """
        Create a combined dataset from a list of datasets.

        Args:
            datasets: A list of datasets. Each dataset consists of x, y (input, label)
                the labels are assumed to be the same across datasets and datasets
                consist of the same underlying data
            is_test: A boolean indicating whether this is a test dataset (i.e. no labels available)
        """
        super().__init__()
        self.split = split

        feature_list = []
        for run_id in runs:
            run_feats = np.float32(np.load(load_wandb_file(
                "{}_embeddings.npy".format(split), run_id, save_dir)))
            if len(run_feats.shape) == 1:  # fix single dim arrays for concatenating
                run_feats = run_feats.reshape(-1, 1)
            feature_list.append(run_feats)

        self.features = np.concatenate(feature_list, axis=1)
        self.dim = self.features.shape[1]

        if split != "test":
            self.load_labels()
            assert len(self.labels) == len(self.features)

    def load_labels(self):
        filename = "./twitter-datasets/full_" + self.split + ".csv"
        self.labels = torch.tensor(pd.read_csv(
            filename)["labels"], dtype=torch.float32)

    def __getitem__(self, index):
        if self.split == "test":
            return self.features[index]
        else:
            return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)
