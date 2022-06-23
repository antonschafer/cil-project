import torch
from transformers import AutoTokenizer
import yaml

from modeling.base_dataset import BaseDataset
from modeling.base_module import BaseModule
from modeling.base_testdataset import BaseTestDataset
from modeling.twitter_roberta.twitter_roberta_module import TwitterRobertaModule

MODELS = {
    "base": {
        "model_name": "bert-base-uncased",
        "tokenizer_name": "bert-base-uncased",
        "module": BaseModule
    },
    "twitter_roberta": { # cardiffnlp/twitter-roberta-base-sentiment-latest model.
        # To use: 
        #   1. install git-lfs 
        #       - on Euler run: env2lmod, module load git-lfs
        #   2. clone model repo into huggingface_repos
        #       - "git clone https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest"
        #   3. cd to huggingface_repos/twitter-roberta-base-sentiment-latest and run "git lfs pull"
        "model_name": "./huggingface_repos/twitter-roberta-base-sentiment-latest",
        "tokenizer_name": "./huggingface_repos/twitter-roberta-base-sentiment-latest",
        "module": TwitterRobertaModule,
    },
    "twitter_xlm_roberta": {
        "model_name": "./huggingface_repos/twitter-xlm-roberta-base-sentiment",
        "toeknizer_name": "./huggingface_repos/twitter-xlm-roberta-base-sentiment",
        "module": TwitterRobertaModule,
    }
}


def get_bert_config(args):
    # read args
    if args.config_path == '':
        config = vars(args)
    else:
        with open(args.config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    # retrieve model info
    config["model_name"] = MODELS[config["model"]]["model_name"]
    config["tokenizer_name"] = MODELS[config["model"]]["tokenizer_name"]
    module = MODELS[config["model"]]["module"]

    config["gpus"] = 1 if torch.cuda.is_available() else 0
    
    print('Config:')
    print(config)

    return config, module


def get_base_datasets(config):
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    data = BaseDataset(tokenizer=tokenizer, full_data=config['full_data'])
    test_data = BaseTestDataset(tokenizer=tokenizer)

    n_val = round(len(data) * 0.1)  # val size always 0.1 to keep train val split fixed. TODO messy if sometimes using full data and sometimes not
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))  # Fix train and val split
    
    return train_data, val_data, test_data
