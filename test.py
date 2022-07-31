import argparse
import os
import warnings
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import wandb
from utils import get_base_datasets, get_bert_config, load_wandb_checkpoint

TEST_BATCH_SIZE = 64


def run_and_save_val(trainer, model, dataset, ckpt_path, split_name):
    dataloader = DataLoader(
        dataset, batch_size=TEST_BATCH_SIZE, num_workers=1, pin_memory=True
    )
    outputs = trainer.predict(model, dataloader, ckpt_path=ckpt_path)
    preds, labels, loss = model.aggregate_outputs(outputs)
    correct_preds = (preds > 0.5) == labels
    np.save(os.path.join(wandb.run.dir, "{}_preds.npy".format(split_name)), preds)
    np.save(
        os.path.join(wandb.run.dir, "{}_correct_preds.npy".format(split_name)),
        correct_preds,
    )
    wandb.run.summary["best_{}_loss".format(split_name)] = loss
    wandb.run.summary["best_{}_acc".format(split_name)] = correct_preds.mean()

    print("Model Checkpoint Classification Report on {} data:".format(split_name))
    print(classification_report(labels, preds > 0.5, zero_division=0, digits=4))


def run_eval(model, ckpt_path, train_ensemble_set, val_set, val_final_set, test_set):
    trainer = pl.Trainer(accelerator="auto", max_epochs=1)

    run_and_save_val(trainer, model, train_ensemble_set, ckpt_path, "train_ensemble")
    run_and_save_val(trainer, model, val_set, ckpt_path, "val")
    run_and_save_val(trainer, model, val_final_set, ckpt_path, "val_final")

    test_loader = DataLoader(
        test_set, batch_size=TEST_BATCH_SIZE, num_workers=1, pin_memory=True
    )
    trainer.test(model, test_loader, ckpt_path=ckpt_path)


def test(config, module):
    model = module(config=config)

    ckpt_path = None
    if config["save_path"] is not None:
        ckpt_path = config["save_path"]
    if config["run_id"] is not None:
        ckpt_path = load_wandb_checkpoint(config["run_id"], save_dir=config["save_dir"])

    if config["save_to_wandb"]:
        wandb.init(
            project="twitter-sentiment-analysis",
            dir=config["save_dir"],
            id=config["run_id"],
            resume="must",
        )
    else:
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(dir=config["save_dir"])

    _, train_ensemble_set, val_set, val_final_set, test_set = get_base_datasets(config)
    run_eval(model, ckpt_path, train_ensemble_set, val_set, val_final_set, test_set)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--save_to_wandb", action="store_true", help="save logs to wandb run"
    )
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join("/cluster/scratch", os.environ["USER"]),
        help="where to save and cache files",
    )

    args = parser.parse_args()
    config, module = get_bert_config(args)

    # force to always run validation on full dataset so uploaded / logged results stay consistent
    # if need to run quick testing without validation, run with --test_only
    config["full_data"] = True

    # check config
    if config["save_path"] is not None and config["run_id"] is not None:
        raise ValueError("Can either restore from run_id or from file, not both")
    if config["run_id"] is None and config["save_to_wandb"]:
        raise ValueError("Can only save to wandb if restoring from run_id")

    # we are using 1 worker and that's ok
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    test(config, module)
