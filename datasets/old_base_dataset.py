from torch.utils.data import Dataset
import torch


class BaseDataset(Dataset):
    def __init__(self, tokenizer, full_data=True, transform=None):
        super().__init__()
        self.full_data = full_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.load_data()
        self.data, self.labels = self.preprocess_data()

    def load_data(self):
        def read_txt(filename):
            with open(filename) as f:
                data = f.readlines()
            return set(data)

        if self.full_data:
            self.data_neg = read_txt("./twitter-datasets/train_neg_full.txt")
            self.data_pos = read_txt("./twitter-datasets/train_pos_full.txt")
        else:
            self.data_neg = read_txt("./twitter-datasets/train_neg.txt")
            self.data_pos = read_txt("./twitter-datasets/train_pos.txt")

    def preprocess_data(self):

        # apply transform
        if self.transform is not None:
            print("Applying data transform...")
            self.data_pos = list(map(self.transform, self.data_pos))
            self.data_neg = list(map(self.transform, self.data_neg))

        # tokenize
        print("Tokenizing data...")
        tokens_pos = self.tokenizer(
            list(self.data_pos),
            padding="max_length",
            max_length=104,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        tokens_neg = self.tokenizer(
            list(self.data_neg),
            padding="max_length",
            max_length=104,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        all_tokens = torch.cat((tokens_pos, tokens_neg))
        labels = torch.cat(
            (torch.ones((len(self.data_pos))), torch.zeros((len(self.data_neg))))
        )
        labels_vector = torch.zeros((labels.shape[0], 2))
        labels_vector[range(labels_vector.shape[0]), labels.long()] = 1
        return all_tokens, labels_vector

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
