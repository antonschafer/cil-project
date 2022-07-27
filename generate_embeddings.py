import argparse
import os
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from test import TEST_BATCH_SIZE
<<<<<<< HEAD
from utils import get_base_datasets, get_bert_config, load_wandb_checkpoint, device
from models.base_module import BaseModule
=======
from utils import get_base_datasets, get_bert_config, device

>>>>>>> 07a61c086b95bc42f879cfba4163c3e37f25951b

def get_embeddings(model, dataset, has_labels, use_preds):
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE)
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x = batch[0] if has_labels else batch
            x = x.to(device)
<<<<<<< HEAD
            out = model(x)
            
            # not using pooler output as "were not initialized from the model checkpoint at cardiffnlp/twitter-xlm-roberta-base and are newly initialized"
            if "last_hidden_states" in out.keys():
                cls_embedding = out["last_hidden_states"][:, 0]
            else:
                cls_embedding = out["hidden_states"][-1][:, -1]
            embeddings.append(cls_embedding.detach().cpu())
=======
            out = model(x, output_hidden_states=True)
            if use_preds:
                embeddings.append(out.logits.softmax(dim=-1).detach().cpu())
            else:
                hidden_states = out["hidden_states"]
                last_4_cls = torch.cat([hidden_states[i][:, 0] for i in [-4, -3, -2, -1]], dim=-1)
                embeddings.append(last_4_cls.detach().cpu())
>>>>>>> 07a61c086b95bc42f879cfba4163c3e37f25951b
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

<<<<<<< HEAD
    ckpt_path = None
    if config["run_id"] is not None:
        ckpt_path = load_wandb_checkpoint(
            config["run_id"], save_dir=config["save_dir"])

    else:
        print("Using pre trained models")

    if isinstance(model,BaseModule):
        if ckpt_path is not None:
            model.load_ckpt(ckpt_path)
        model = model.model

    _, val_set, val_final_set, test_set = get_base_datasets(config)

    val_embeddings = get_embeddings(model, val_set, has_labels=True)
    np.save(os.path.join(wandb.run.dir, "val_embeddings.npy"), val_embeddings)

    val_final_embeddings = get_embeddings(
        model, val_final_set, has_labels=True)
    np.save(os.path.join(wandb.run.dir, "val_final_embeddings.npy"),
            val_final_embeddings)

    test_embeddings = get_embeddings(model, test_set, has_labels=False)
    np.save(os.path.join(wandb.run.dir, "test_embeddings.npy"), test_embeddings)
=======
    _, train_ensemble_set, val_set, val_final_set, test_set = get_base_datasets(config)

    train_ensemble_embeddings = get_embeddings(model, train_ensemble_set, has_labels=True, use_preds=config["use_preds"])
    np.save(os.path.join(wandb.run.dir, "train_ensemble_preds.npy"), train_ensemble_embeddings)

    val_embeddings = get_embeddings(model, val_set, has_labels=True, use_preds=config["use_preds"])
    np.save(os.path.join(wandb.run.dir, "val_preds.npy"), val_embeddings)

    val_final_embeddings = get_embeddings(
        model, val_final_set, has_labels=True, use_preds=config["use_preds"])
    np.save(os.path.join(wandb.run.dir, "val_final_preds.npy"),
            val_final_embeddings)

    test_embeddings = get_embeddings(model, test_set, has_labels=False, use_preds=config["use_preds"])
    np.save(os.path.join(wandb.run.dir, "test_preds.npy"), test_embeddings)
>>>>>>> 07a61c086b95bc42f879cfba4163c3e37f25951b


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--save_dir', type=str, default=os.path.join(
        "/cluster/scratch", os.environ["USER"]), help="where to save and cache files")
    parser.add_argument('--save_to_wandb', action='store_true',
                        help="upload to wandb")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_data_size', type=float, default=1.)
    parser.add_argument('--run_id', type=str, default=None)

    parser.add_argument('--use_preds', action='store_true',
                        help="use predictions instead of hidden state (e.g. for T/N/F model")

    args = parser.parse_args()
    args.config_path = ""  # just for get_bert_config # TODO solve cleaner
    config, module = get_bert_config(args)

    # always generate for full dataset
    config["full_data"] = True

    config["output_hidden_states"] = True
    # we are using 1 worker and that's ok
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    save_embeddings(config, module)
