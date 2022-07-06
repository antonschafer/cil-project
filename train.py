import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from utils import get_base_datasets, get_bert_config, compute_metrics

from pytorch_lightning.loggers import WandbLogger
import wandb

DEBUG_TRAINER_ARGS = {"limit_train_batches": 10,
                      "limit_val_batches": 5}


def train(config, module):
    model = module(config=config)

    wandb_logger = WandbLogger(
        project="twitter-sentiment-analysis", name=config["run_name"], offline=True)

    callbacks = [EarlyStopping(monitor="val_loss", mode="min"),
                 ModelCheckpoint(monitor='val_loss', dirpath=wandb.run.dir, filename="model")]

    extra_args = DEBUG_TRAINER_ARGS if config["debug"] else {}
    trainer = pl.Trainer(max_epochs=config['nepochs'], gpus=config["gpus"], callbacks=callbacks,
                         check_val_every_n_epoch=config['val_freq'], gradient_clip_val=1, logger=wandb_logger,
                         **extra_args)

    train_set, val_set, _, test_set = get_base_datasets(config)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=False,
                              num_workers=1)
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, drop_last=False, pin_memory=False,
                             num_workers=4)


    # TODO also run on val_final set (make sure to log with metrics with proper name, not just val_acc)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    try:
        compute_metrics(model, val_set, config['batch_size'], config["run_name"])
    except:
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='')  # TODO needed?
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--run_name', type=str, default=None)

    parser.add_argument('--nepochs', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--full_data', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="only run a few batches")

    args = parser.parse_args()
    config, module = get_bert_config(args)

    train(config, module)
