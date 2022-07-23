import argparse
import os
import warnings
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from test import TEST_BATCH_SIZE
from utils import get_base_datasets, get_bert_config, load_wandb_checkpoint, device


def get_embeddings(model, dataset, has_labels):
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE)
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x = batch[0] if has_labels else batch
            x = x.to(device)
            out = model(x)
            # not using pooler output as "were not initialized from the model checkpoint at cardiffnlp/twitter-xlm-roberta-base and are newly initialized"
            cls_embedding = out["last_hidden_state"][:, 0]
            embeddings.append(cls_embedding.detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


def save_embeddings(config, module):
    model = module(config)

    if config["save_to_wandb"]:
        wandb.init(project="twitter-sentiment-analysis",
                   dir=config["save_dir"], name=config["run_name"], config=config)
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(name=config["run_name"],
                   dir=config["save_dir"], config=config)

    _, train_ensemble_set, val_set, val_final_set, test_set = get_base_datasets(config)

    train_ensemble_embeddings = get_embeddings(model, train_ensemble_set, has_labels=True)
    np.save(os.path.join(wandb.run.dir, "train_ensemble_preds.npy"), train_ensemble_embeddings)

    val_embeddings = get_embeddings(model, val_set, has_labels=True)
    np.save(os.path.join(wandb.run.dir, "val_preds.npy"), val_embeddings)

    val_final_embeddings = get_embeddings(
        model, val_final_set, has_labels=True)
    np.save(os.path.join(wandb.run.dir, "val_final_preds.npy"),
            val_final_embeddings)

    test_embeddings = get_embeddings(model, test_set, has_labels=False)
    np.save(os.path.join(wandb.run.dir, "test_preds.npy"), test_embeddings)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--save_dir', type=str, default=os.path.join(
        "/cluster/scratch", os.environ["USER"]), help="where to save and cache files")
    parser.add_argument('--save_to_wandb', action='store_true',
                        help="upload to wandb")
    parser.add_argument('--run_name', type=str, default=None)

    args = parser.parse_args()
    args.config_path = ""  # just for get_bert_config # TODO solve cleaner
    config, module = get_bert_config(args)

    # always generate for full dataset
    config["full_data"] = True

    # we are using 1 worker and that's ok
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    save_embeddings(config, module)
