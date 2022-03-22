from abc import ABC

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd


class DataModule(pl.LightningDataModule, ABC):
    def __init__(self):
        super().__init__()
        self.train_neg = pd.read_csv('./../twitter-datasets/train_neg_full.txt', sep='\n', error_bad_lines=False,
                                     header=None)
        self.train_pos = pd.read_csv('./../twitter-datasets/train_pos_full.txt', sep='\n', error_bad_lines=False,
                                     header=None)
        self.test_data = pd.read_csv('./../twitter-datasets/test_data.txt', sep='\n', error_bad_lines=False,
                                     header=None)

    def preprocess_data(self):
        self.train_pos.drop_duplicates(inplace=True)
        self.train_neg.drop_duplicates(inplace=True)

    def train_pos_dataloader(self):
        return DataLoader(self.train_pos, batch_size=32)

    def train_neg_dataloader(self):
        return DataLoader(self.train_neg, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=32)


