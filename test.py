import argparse
import os
from numpy import save
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
from utils import get_base_datasets, get_bert_config, load_wandb_checkpoint


def test(config, module):
    model = module(config=config)

    # restore model weights
    if config["save_path"] is not None:
        model.load_ckpt(config['save_path'])
    if config["run_id"] is not None:
        model.load_ckpt(load_wandb_checkpoint(
            config["run_id"], save_dir=config["save_dir"]))

    if config["save_to_wandb"]:
        # make sure pred files are saved to correct run
        trainer = pl.Trainer(accelerator="auto", logger=WandbLogger(
            id=config["run_id"], save_dir=config["save_dir"], project="twitter-sentiment-analysis", resume="must"))
    else:
        wandb.init()
        os.environ["WANDB_MODE"] = "dryrun"
        trainer = pl.Trainer(accelerator="auto")

    _, val_set, val_final_set, test_set = get_base_datasets(config)
    test_loader = DataLoader(test_set, batch_size=64,
                             num_workers=4, pin_memory=True)
    if config["test_only"]:
        val_loader, val_final_loader = None, None
    else:
        val_loader = DataLoader(val_set, batch_size=64,
                                num_workers=4, pin_memory=True)
        val_final_loader = DataLoader(
            val_final_set, batch_size=64, num_workers=4, pin_memory=True)

    model.run_final_eval(trainer=trainer, val_loader=val_loader,
                         val_final_loader=val_final_loader, test_loader=test_loader, save_preds=config["save_to_wandb"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--save_to_wandb', action='store_true',
                        help="save logs to wandb run")
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join("/cluster/scratch", os.environ["USER"]))

    args = parser.parse_args()
    config, module = get_bert_config(args)

    # force to always run validation on full dataset so uploaded / logged results stay consistent
    # if need to run quick testing without validation, run with --test_only
    config["full_data"] = True

    # check config
    if config["save_path"] is not None and config["run_id"] is not None:
        raise ValueError(
            "Can either restore from run_id or from file, not both")
    if config["run_id"] is None and config["save_to_wandb"]:
        raise ValueError(
            "Can only save to wandb if restoring from run_id")

    test(config, module)
