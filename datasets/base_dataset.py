from torch.utils.data import Dataset
import torch
import pandas as pd

from datasets.version import DATA_VERSION


class BaseDataset(Dataset):
    def __init__(
        self,
        split,
        tokenizer,
        seed=0,
        full_data=True,
        transform=None,
        train_data_size=None,
        pad=True,
        text_with_prompt=False,
    ):
        super().__init__()
        self.split = split
        self.full_data = full_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.seed = seed
        self.train_data_size = train_data_size
        self.pad = pad
        self.text_with_prompt = text_with_prompt
        self.load_data()
        self.data, self.labels = self.preprocess_data()

    def load_data(self):
        filename = (
            "./twitter-datasets/"
            + ("full_" if self.full_data else "small_")
            + self.split
            + "_v{}.csv".format(DATA_VERSION)
        )
        self.df = pd.read_csv(filename)
        if self.train_data_size is not None:
            self.df = self.df.sample(
                int(self.df.shape[0] * self.train_data_size), random_state=self.seed
            ).reset_index(drop=True)

        if self.text_with_prompt:
            self.df["texts"] = self.df["texts"].apply(
                lambda x: "Twitter Sentiment Analysis Examples.\n\n Tweet: {}\n Sentiment: ".format(
                    x
                )
            )

    def preprocess_data(self):

        # apply transform
        self.texts = self.df["texts"]
        if self.transform is not None:
            print("Applying data transform...")
            self.texts = list(map(self.transform, self.texts))

        # tokenize
        print("Tokenizing data...")

        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # self.tokenizer.pad_token = self.tokenizer.eos_token
        all_tokens = self.tokenizer(
            list(self.texts),
            padding="max_length",
            max_length=104,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        labels_vector = torch.zeros((self.df.shape[0], 2))
        labels_vector[range(labels_vector.shape[0]), self.df["labels"]] = 1
        return all_tokens, labels_vector

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
