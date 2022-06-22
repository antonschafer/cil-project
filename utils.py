import torch
from transformers import AutoTokenizer
import yaml

from modeling.base_dataset import BaseDataset
from modeling.base_testdataset import BaseTestDataset

def get_bert_config(args):
    # use same tokenizer as model
    if args.tokenizer_name == "":
        args.tokenizer_name = args.model_name

    config = ""
    if args.config_path == '':
        config = vars(args)
    else:
        with open(args.config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    print('Config:')
    print(config)

    return config


def get_base_datasets(config):
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    data = BaseDataset(tokenizer=tokenizer, full_data=config['full_data'])
    test_data = BaseTestDataset(tokenizer=tokenizer)

    n_val = round(len(data) * config["val_size"])
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))  # Fix train and val split
    
    return train_data, val_data, test_data
