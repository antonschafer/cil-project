import os

os.environ["TRANSFORMERS_CACHE"] = "/cluster/scratch/{}/hugging_cache/".format(os.environ["USER"])
from torch.utils.data import DataLoader
import wandb

from utils import get_base_arg_parser, get_base_datasets, get_bert_config, get_trainer
from test import TEST_BATCH_SIZE, run_eval
import warnings
import numpy as np
import torch

def train(config, module):
    model = module(config=config)
    trainer = get_trainer(config)

    train_set, train_ensemble_set, val_set, val_final_set, test_set = get_base_datasets(config)

    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=1)
    val_loader = DataLoader(
        val_set, batch_size=TEST_BATCH_SIZE, num_workers=1)

    trainer.fit(model, train_loader, val_loader)

    if config["nepochs"] == 0:
        # use model as is if no training, useful to directly evaluate pretrained HF models
        ckpt_path = os.path.join(wandb.run.dir, "model.ckpt")
        trainer.save_checkpoint(ckpt_path)
        assert (not config["debug"]) and (config["full_data"]), "log only if using full data"
    else:
        ckpt_path = trainer.checkpoint_callback.best_model_path

    run_eval(model, ckpt_path=ckpt_path, train_ensemble_set=train_ensemble_set, val_set=val_set, val_final_set=val_final_set, test_set=test_set)


if __name__ == '__main__':

    parser = get_base_arg_parser()
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--log_only', action='store_true', help="only evaluate and log to wandb, skip training. useful for pretrained HF models")

    args = parser.parse_args()
    config, module = get_bert_config(args)

    # we are using 1 worker and that's ok
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    # we don't want to be warned that awndb run dir where checkpoint is saved is not empty
    warnings.filterwarnings("ignore", ".*exists and is not empty*")

    train(config, module)
