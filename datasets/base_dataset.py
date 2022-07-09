from torch.utils.data import Dataset
import torch
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, split, tokenizer, full_data=True, transform=None):
        super().__init__()
        self.split = split
        self.full_data = full_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.load_data()
        self.data, self.labels = self.preprocess_data()

    def load_data(self):
        filename = "./twitter-datasets/" + ("full_" if self.full_data else "small_") + self.split+".csv"
        self.df = pd.read_csv(filename)
        

    def preprocess_data(self):

        # apply transform
        self.texts = self.df["texts"]
        if self.transform is not None:
            print("Applying data transform...")
            self.texts = list(map(self.transform, self.texts))
            

        # tokenize
        print("Tokenizing data...")
        all_tokens = self.tokenizer(list(self.texts), padding='max_length',
                                    max_length=104, truncation=True, return_tensors="pt")['input_ids']

        labels_vector = torch.zeros((self.df.shape[0], 2))
        labels_vector[range(labels_vector.shape[0]), self.df["labels"]] = 1
        return all_tokens, labels_vector

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
