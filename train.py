import argparse
from gc import callbacks
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from .base_module import BaseModule

def train(config):
    model = BaseModule(config=config)
    callbacks = [EarlyStopping(monitor="val_loss", mode="min"),
                    ModelCheckpoint(monitor='val_loss', dirpath=config['save_path'])]
    trainer = pl.Trainer(max_epochs=config['nepochs'],gpus=1,callbacks=callbacks,
                            check_val_every_n_epoch=config['val_freq'],gradient_clip_val=1)
    train_set, val_set = ...
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=4)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--nepochs', type=int, default=1)
    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()
    if args.config_path == '':
        config = vars(args)
    else:
        with open(args.config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    print('Config:')
    print(config)
    train(config)