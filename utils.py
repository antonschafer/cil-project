import torch
from transformers import AutoTokenizer
import os
import yaml
import time
import pandas as pd
from datasets.base_dataset import BaseDataset
from models.base_module import BaseModule
from datasets.base_testdataset import BaseTestDataset
from models.three_class_module import ThreeClassModule
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# To use offline models on euler, you need to download the model and tokenizer:
#   1. install git-lfs
#       - on Euler run: env2lmod, module load git-lfs
#   2. clone model repo into huggingface_repos
#       - e.g. "git clone https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest"
#   3. cd to huggingface_repos/twitter-roberta-base-sentiment-latest and run "git lfs pull"

MODELS = {
    "base": {
        "model_name": "bert-base-uncased",
        "tokenizer_name": "bert-base-uncased",
        "module": BaseModule,
        "data_transform": None,
    },
    "twitter_roberta": {  # cardiffnlp/twitter-roberta-base-sentiment-latest
        "model_name": "./huggingface_repos/twitter-roberta-base-sentiment-latest",
        "tokenizer_name": "./huggingface_repos/twitter-roberta-base-sentiment-latest",
        "module": ThreeClassModule,
        "data_transform": lambda x: x.replace("<user>", "@user"),
    },
    "twitter_xlm_roberta": {  # cardiffnlp/twitter-xlm-roberta-base-sentiment
        "model_name": "./huggingface_repos/twitter-xlm-roberta-base-sentiment",
        "tokenizer_name": "./huggingface_repos/twitter-xlm-roberta-base-sentiment",
        "module": ThreeClassModule,
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

    # retrieve model info
    config["model_name"] = MODELS[config["model"]]["model_name"]
    config["tokenizer_name"] = MODELS[config["model"]]["tokenizer_name"]
    module = MODELS[config["model"]]["module"]

    config["gpus"] = 1 if torch.cuda.is_available() else 0

    print('Config:')
    print(config)

    return config, module


def get_base_datasets(config, test=True, train_val=True):
    # TODO cache tokenized datasets, make sure that not confused when differnet tokenizer, full_data option or transform used
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    data_transform = MODELS[config['model']]['data_transform']

    if train_val:
        data = BaseDataset(
            tokenizer=tokenizer, full_data=config['full_data'], transform=data_transform)

        # split train, val, val_final: 1.1M, 100K, 50K
        n_val = round(len(data) * 0.08)
        n_val_final = round(len(data) * 0.04)
        n_train = len(data) - n_val - n_val_final

        # fix split with random seed. Note that splits might still be different when using full dataset vs small dataset
        train_data, val_data, val_final_data = torch.utils.data.random_split(data, [n_train, n_val, n_val_final],
                                                                             generator=torch.Generator().manual_seed(
                                                                                 42))
    else:
        train_data, val_data, val_final_data = None, None, None

    if test:
        test_data = BaseTestDataset(
            tokenizer=tokenizer, transform=data_transform)
    else:
        test_data = None

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
    pd.DataFrame(torch.Tensor(val_set.dataset.labels.argmax(axis=1).tolist()) == torch.Tensor(preds)).to_csv('statistics/' + name + '.csv', header=None, index=False)
    merge_metrics()

def merge_metrics():
    df = pd.DataFrame(columns=['model1', 'model2', 'model3', 'coverage'])
    df.to_csv(os.path.join('statistics', 'coverage.csv'))

    for i, filename in enumerate(os.listdir('statistics')):
        if filename == 'coverage.csv':
            continue

        cov = pd.DataFrame([[filename, None, None, pd.read_csv(os.path.join('statistics', filename)).mean()[0]]])
        cov.to_csv(os.path.join('statistics', 'coverage.csv'), header=False, mode='a')

        for j, filename2 in enumerate(os.listdir('statistics')):
            if filename2 == 'coverage.csv' or j <= i:
                continue

            comb = pd.read_csv(os.path.join('statistics', filename)).iloc[:, 0] | pd.read_csv(os.path.join('statistics', filename2)).iloc[:, 0]
            cov = pd.DataFrame([[filename, filename2, None, comb.mean()]])
            cov.to_csv(os.path.join('statistics', 'coverage.csv'), header=False, mode='a')

            for k, filename3 in enumerate(os.listdir('statistics')):
                if filename3 == 'coverage.csv' or filename == filename3 or k <= j:
                    continue

                combs = comb | pd.read_csv(os.path.join('statistics', filename3)).iloc[:, 0]
                cov = pd.DataFrame([[filename, filename2, filename3, combs.mean()]])
                cov.to_csv(os.path.join('statistics', 'coverage.csv'), header=False, mode='a')





