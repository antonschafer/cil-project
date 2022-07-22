import argparse
import pytorch_lightning as pl
import hashlib
import dill  # can pickle lambdas
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import os
import wandb
import yaml
import pandas as pd
from datasets.base_dataset import BaseDataset
from datasets.base_testdataset import BaseTestDataset
from datasets.version import DATA_VERSION
from models.binary_hf_module import BinaryHFModule
from models.three_class_hf_module import ThreeClassHFModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
import wandb

DEBUG_TRAINER_ARGS = {"limit_train_batches": 10,
                      "limit_val_batches": 5}

WANDB_PROJECT_PATH = "cil-biggoodteam/twitter-sentiment-analysis/"


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
        "data_transform": lambda x: x.replace("<user>", "@user").replace("<url>", "http"),
    },
    "twitter_roberta_nonlatest": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment",
        "tokenizer_name": "cardiffnlp/twitter-roberta-base-sentiment",
        "module": ThreeClassHFModule,
        "data_transform": lambda x: x.replace("<user>", "@user").replace("<url>", "http"),
    },
    "twitter_xlm_roberta": {
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "tokenizer_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "module": ThreeClassHFModule,
        "data_transform": lambda x: x.replace("<user>", "@user").replace("<url>", "http"),
    },
    "distilbert_emotion": {
        "model_name": "bhadresh-savani/distilbert-base-uncased-emotion",
        "tokenizer_name": "bhadresh-savani/distilbert-base-uncased-emotion",
        "module": BinaryHFModule,
        "data_transform": None
    },

    # --------------------------------------------------------------------------------
    # Models only for generating embeddings
    # --------------------------------------------------------------------------------
    "twitter_roberta_embeddings": {
        "model_name": "cardiffnlp/twitter-roberta-base",
        "tokenizer_name": "cardiffnlp/twitter-roberta-base",
        "module": lambda _: AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base"),
        "data_transform": lambda x: x.replace("<user>", "@user").replace("<url>", "http"),
    },
    "twitter_roberta_pos_neut_neg_preds": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment",
        "tokenizer_name": "cardiffnlp/twitter-roberta-base-sentiment",
        "module": lambda _: AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment"),
        "data_transform": lambda x: x.replace("<user>", "@user").replace("<url>", "http"),
    },
    # --------------------------------------------------------------------------------
    # Only for tokenization
    # --------------------------------------------------------------------------------
    "gpt2_tokenizer": {
        "model_name": None,
        "tokenizer_name": "gpt2",
        "module": None,
        "data_transform": None
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
        config["save_dir"] = os.path.join(
            "/cluster/scratch", os.environ["USER"])
    
    # add data version
    config["DATA_VERSION"] = DATA_VERSION

    # retrieve model info
    config["model_name"] = MODELS[config["model"]]["model_name"]
    config["tokenizer_name"] = MODELS[config["model"]]["tokenizer_name"]
    module = MODELS[config["model"]]["module"]


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

    # set defaults
    if "train_data_size" not in config:
        config["train_data_size"] = None
    if "seed" not in config:
        config["seed"] = 0

    # check if can load from cache
    option_str = "_".join(
        [config["tokenizer_name"].split("/")[-1], str(config["full_data"]), str(function_to_hash(data_transform)), str(config["seed"]), str(config["train_data_size"]), "v" + str(DATA_VERSION)])
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
        train_data = BaseDataset(split="train", tokenizer=tokenizer,
                                 full_data=config['full_data'], seed=config['seed'], transform=data_transform, train_data_size=config['train_data_size'])
        train_ensemble_data = BaseDataset(split="train_ensemble", tokenizer=tokenizer,
                                 full_data=config['full_data'], seed=config['seed'], transform=data_transform, train_data_size=config['train_data_size'])
        val_data = BaseDataset(split="val", tokenizer=tokenizer,
                               full_data=config['full_data'], transform=data_transform)
        val_final_data = BaseDataset(
            split="val_final", tokenizer=tokenizer, full_data=config['full_data'], transform=data_transform)

        test_data = BaseTestDataset(
            tokenizer=tokenizer, transform=data_transform)

        # save to cache
        print("Saving datasets to cache:", cache_file)
        write_pickle(
            [train_data, train_ensemble_data, val_data, val_final_data, test_data], cache_file)

        return train_data, train_ensemble_data, val_data, val_final_data, test_data




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


def get_trainer(config):
    os.makedirs(config["save_dir"], exist_ok=True)
    wandb_logger = WandbLogger(
        project="twitter-sentiment-analysis", name=config["run_name"], save_dir=config["save_dir"])

    callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=config["es_patience"]),
                 ModelCheckpoint(monitor='val_loss', dirpath=wandb.run.dir, filename="model")]

    extra_args = DEBUG_TRAINER_ARGS if config["debug"] else {}
    trainer = pl.Trainer(max_epochs=config['nepochs'], accelerator="auto", callbacks=callbacks,
                         val_check_interval=config['val_check_interval'], gradient_clip_val=1, logger=wandb_logger,
                         accumulate_grad_batches=config['accumulate_grad_batches'],
                         **extra_args)
    return trainer


def get_base_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='')  # TODO needed?
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join("/cluster/scratch", os.environ["USER"]))

    parser.add_argument('--nepochs', type=int, default=1)
    # to validate only once per epoch, use 1.0 (not 1)
    parser.add_argument('--val_check_interval', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_data_size', type=float, default=None)

    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--es_patience', type=int, default=3,
                        help="early stopping patience")

    parser.add_argument('--full_data', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="only run a few batches")

    return parser


def load_wandb_checkpoint(run_id, save_dir):
    """
    Load checkpoint file from wandb
    """
    return load_wandb_file("model.ckpt", run_id, save_dir)


def load_wandb_file(fname, run_id, save_dir):
    """
    Load file from wandb
    """
    print("Loading {} from wandb...".format(fname))
    # download file
    run_cache_dir = os.path.join(save_dir, "cache", run_id)
    os.makedirs(run_cache_dir, exist_ok=True)
    return wandb.restore(
        fname, run_path=WANDB_PROJECT_PATH + run_id, root=run_cache_dir).name

