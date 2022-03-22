import pandas as pd
from torch.utils.data import Dataset
import torch

class BaseDataset(Dataset):
    def __init__(self,tokenizer,full_data=True):
        super().__init__()
        self.full_data = full_data
        self.tokenizer = tokenizer
        self.load_data()
        self.data, self.labels = self.preprocess_data()

    def load_data(self):
        def read_txt(filename):
            with open(filename) as f:
                data = f.readlines()
            return set(data)

        if self.full_data:
            self.data_neg = read_txt('./twitter-datasets/train_neg_full.txt')
            self.data_pos = read_txt('./twitter-datasets/train_pos_full.txt')
        else:
            self.data_neg = read_txt('./twitter-datasets/train_neg.txt')
            self.data_pos = read_txt('./twitter-datasets/train_pos.txt')

    def preprocess_data(self):
        tokens_pos = self.tokenizer(self.data_pos, padding=True, truncation=True, return_tensors="pt")['input_ids']
        tokens_neg = self.tokenizer(self.data_neg, padding=True, truncation=True, return_tensors="pt")['input_ids']
        all_tokens = torch.cat((tokens_pos,tokens_neg))
        labels = torch.cat((torch.ones((len(self.data_pos))),torch.zeros((len(self.data_neg)))))
        return all_tokens,labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    
