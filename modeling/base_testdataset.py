from torch.utils.data import Dataset
import torch

class BaseTestDataset(Dataset):
    def __init__(self,tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.load_data()
        self.data = self.preprocess_data()
   
    def load_data(self):
        def read_txt(filename):
            with open(filename) as f:
                data = f.readlines()
            return list(data)
        
        self.test_data = read_txt('twitter-datasets/test_data.txt')
    
    def preprocess_data(self):
        tokens_pos = self.tokenizer(list(self.test_data), padding='max_length', max_length=104, truncation=True, return_tensors="pt")['input_ids']
        #tokens_pos = self.tokenizer(list(self.test_data), padding=True, truncation=True, return_tensors="pt")['input_ids']
        return tokens_pos

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
