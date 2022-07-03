import hashlib
import dill  # can pickle lambdas
import torch
from transformers import AutoTokenizer
import os
import yaml
import time
import pandas as pd
from datasets.base_dataset import BaseDataset
from models.base_module import BaseModule
from datasets.base_testdataset import BaseTestDataset
from models.binary_hf_module import BinaryHFModule
from models.three_class_hf_module import ThreeClassHFModule
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

MODELS = {
    "base": {
        "model_name": "bert-base-uncased",
        "tokenizer_name": "bert-base-uncased",
        "module": BinaryHFModule,
        "data_transform": None,
    },
    "twitter_roberta": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "tokenizer_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "module": ThreeClassHFModule,
        "data_transform": lambda x: x.replace("<user>", "@user"),
    },
    "twitter_xlm_roberta": {
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "tokenizer_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "module": ThreeClassHFModule,
        "data_transform": lambda x: x.replace("<user>", "@user"),
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

    if config["save_dir"] == "":
        config["save_dir"] = os.path.join("/cluster/scratch", os.environ["USER"])
    
    config["debug"] = args.debug

    # retrieve model info
    config["model_name"] = MODELS[config["model"]]["model_name"]
    config["tokenizer_name"] = MODELS[config["model"]]["tokenizer_name"]
    module = MODELS[config["model"]]["module"]

    config["gpus"] = int(torch.cuda.is_available())


    print('Config:')
    print(config)

    return config, module


def load_pickle(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def write_pickle(data, path):
    with open(path, 'wb') as f:
        dill.dump(data, f)


def my_hash(b):
    return


def function_to_hash(func):
    if "lambda" in str(func):
        # bytecode instructions
        b = func.__code__.co_code
    else:
        b = str(func).encode("utf-8")
    return int(hashlib.sha256(b).hexdigest(), 16) % 10 ** 12



def get_base_datasets(config):
    data_transform = MODELS[config['model']]['data_transform']

    # check if can load from cache
    option_str = "_".join(
        [config["tokenizer_name"].split("/")[-1], str(config["full_data"]), str(function_to_hash(data_transform))])
    cache_dir = os.path.join(config["save_dir"], "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, option_str + ".pkl")
    if os.path.exists(cache_file):
        print("Loading datasets from cache:", cache_file)
        return load_pickle(cache_file)
    else:
        print("Building datasets...")

        # build datasets
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        train_data = BaseDataset(split="train",tokenizer=tokenizer, full_data=config['full_data'], transform=data_transform)
        val_data = BaseDataset(split="val",tokenizer=tokenizer, full_data=config['full_data'], transform=data_transform)
        val_final_data = BaseDataset(split="val_final",tokenizer=tokenizer, full_data=config['full_data'], transform=data_transform)

        test_data = BaseTestDataset(
            tokenizer=tokenizer, transform=data_transform)

        # save to cache
        print("Saving datasets to cache:", cache_file)
        write_pickle(
            [train_data, val_data, val_final_data, test_data], cache_file)

        return train_data, val_data, val_final_data, test_data


def compute_metrics(model, val_set, batch_size, name):
    model = model.eval().to(device)
    t = 0
    data = val_set.dataset.data.to(device)
    preds = []
    while t < data.shape[0]:
        preds += model(data[t: t + batch_size]).argmax(axis=1).tolist()
        t += batch_size
    if name is None:
        name = str(time.time())
    pd.DataFrame(torch.Tensor(val_set.dataset.labels.argmax(axis=1).tolist()) == torch.Tensor(
        preds)).to_csv('statistics/' + str(name) + '.csv', header=None, index=False)
    merge_metrics()


def merge_metrics():
    df = pd.DataFrame(columns=['model1', 'model2', 'model3', 'coverage'])
    df.to_csv(os.path.join('statistics', 'coverage.csv'))

    for i, filename in enumerate(os.listdir('statistics')):
        if filename == 'coverage.csv':
            continue

        cov = pd.DataFrame([[filename, None, None, pd.read_csv(
            os.path.join('statistics', filename)).mean()[0]]])
        cov.to_csv(os.path.join('statistics', 'coverage.csv'),
                   header=False, mode='a')

        for j, filename2 in enumerate(os.listdir('statistics')):
            if filename2 == 'coverage.csv' or j <= i:
                continue

            comb = pd.read_csv(os.path.join('statistics', filename)).iloc[:, 0] | pd.read_csv(
                os.path.join('statistics', filename2)).iloc[:, 0]
            cov = pd.DataFrame([[filename, filename2, None, comb.mean()]])
            cov.to_csv(os.path.join('statistics', 'coverage.csv'),
                       header=False, mode='a')

            for k, filename3 in enumerate(os.listdir('statistics')):
                if filename3 == 'coverage.csv' or filename == filename3 or k <= j:
                    continue

                combs = comb | pd.read_csv(os.path.join(
                    'statistics', filename3)).iloc[:, 0]
                cov = pd.DataFrame(
                    [[filename, filename2, filename3, combs.mean()]])
                cov.to_csv(os.path.join('statistics', 'coverage.csv'),
                           header=False, mode='a')
