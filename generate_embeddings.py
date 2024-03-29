import argparse
import os
import warnings

os.environ["TRANSFORMERS_CACHE"] = "/cluster/scratch/{}/hugging_cache/".format(
    os.environ["USER"]
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from test import TEST_BATCH_SIZE
from utils import get_base_datasets, get_bert_config, device
from models.sharded_module import ShardedBinaryHFModule


def get_embeddings(model, dataset, has_labels, use_preds):
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE)
    embeddings = []
    with torch.no_grad():
        if isinstance(model, ShardedBinaryHFModule):
            with torch.cuda.amp.autocast():
                model = model.model
                model.eval()
                for batch in tqdm(dataloader):
                    x = batch[0] if has_labels else batch
                    x = x.to(device)
                    out = model(x, output_hidden_states=True)
                    if use_preds:
                        embeddings.append(out.logits.softmax(dim=-1).detach().cpu())
                    else:
                        hidden_states = out["hidden_states"]
                        last_4_cls = torch.cat(
                            [hidden_states[i][:, 0] for i in [-4, -3, -2, -1]], dim=-1
                        )
                        embeddings.append(last_4_cls.detach().cpu())
        else:
            model = model.model
            model.eval()
            for batch in tqdm(dataloader):
                x = batch[0] if has_labels else batch
                x = x.to(device)
                out = model(x, output_hidden_states=True)
                if use_preds:
                    embeddings.append(out.logits.softmax(dim=-1).detach().cpu())
                else:
                    hidden_states = out["hidden_states"]
                    last_4_cls = torch.cat(
                        [hidden_states[i][:, 0] for i in [-4, -3, -2, -1]], dim=-1
                    )
                    embeddings.append(last_4_cls.detach().cpu())

    return torch.cat(embeddings, dim=0).numpy()


def save_embeddings(config, module):

    if config["save_to_wandb"]:
        wandb.init(
            project="twitter-sentiment-analysis",
            dir=config["save_dir"],
            name=config["run_name"],
            config=config,
        )
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(name=config["run_name"], dir=config["save_dir"], config=config)

    _, train_ensemble_set, val_set, val_final_set, test_set = get_base_datasets(config)

    model = module(config)

    train_ensemble_embeddings = get_embeddings(
        model, train_ensemble_set, has_labels=True, use_preds=config["use_preds"]
    )
    np.save(
        os.path.join(wandb.run.dir, "train_ensemble_preds.npy"),
        train_ensemble_embeddings,
    )

    val_embeddings = get_embeddings(
        model, val_set, has_labels=True, use_preds=config["use_preds"]
    )
    np.save(os.path.join(wandb.run.dir, "val_preds.npy"), val_embeddings)

    val_final_embeddings = get_embeddings(
        model, val_final_set, has_labels=True, use_preds=config["use_preds"]
    )
    np.save(os.path.join(wandb.run.dir, "val_final_preds.npy"), val_final_embeddings)

    test_embeddings = get_embeddings(
        model, test_set, has_labels=False, use_preds=config["use_preds"]
    )
    np.save(os.path.join(wandb.run.dir, "test_preds.npy"), test_embeddings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="base")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join("/cluster/scratch", os.environ["USER"]),
        help="where to save and cache files",
    )
    parser.add_argument("--save_to_wandb", action="store_true", help="upload to wandb")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument(
        "--use_preds",
        action="store_true",
        help="use predictions instead of hidden state (e.g. for T/N/F model",
    )

    args = parser.parse_args()
    args.config_path = ""  # just for get_bert_config # TODO solve cleaner
    config, module = get_bert_config(args)

    # always generate for full dataset
    config["full_data"] = True

    # we are using 1 worker and that's ok
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    save_embeddings(config, module)
