from torch.utils.data import Dataset


class BaseTestDataset(Dataset):
    def __init__(self, tokenizer, transform=None, pad=True, text_with_prompt=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.transform = transform
        self.pad = pad
        self.text_with_prompt = text_with_prompt
        self.load_data()
        self.data = self.preprocess_data()

    def load_data(self):
        def read_txt(filename):
            with open(filename) as f:
                data = f.read().split("\n")
                data = [x for x in data if x != ""]
            # drop indices
            return [",".join(x.split(",")[1:]) for x in data]

        self.test_data = read_txt("twitter-datasets/test_data.txt")
        if self.text_with_prompt:
            self.test_data = [
                "Twitter Sentiment Analysis Examples.\n\n Tweet: {}\n Sentiment: ".format(
                    x
                )
                for x in self.test_data
            ]

    def preprocess_data(self):
        # apply transform
        if self.transform is not None:
            print("Applying data transform...")
            self.test_data = list(map(self.transform, self.test_data))

        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # tokenize
        print("Tokenizing data...")
        return self.tokenizer(
            list(self.test_data),
            padding="max_length" if self.pad else False,
            max_length=104,
            truncation=True,
            return_tensors="pt" if self.pad else None,
        )["input_ids"]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
