import argparse
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modeling import *
import torch


def train(config):
    model = BaseModule(config=config)
    callbacks = [EarlyStopping(monitor="val_loss", mode="min"),
                 ModelCheckpoint(monitor='val_loss', dirpath=config['save_path'],filename="model.ckpt")]
    trainer = pl.Trainer(max_epochs=config['nepochs'], gpus=1, callbacks=callbacks,
                         check_val_every_n_epoch=config['val_freq'], gradient_clip_val=1)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    dataset = BaseDataset(tokenizer=tokenizer, full_data=config['full_data'])

    train_set, val_set = torch.utils.data.random_split(dataset, [
        round(len(dataset) * (1 - config['val_size'])), round(len(dataset) * config['val_size'])])
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=False,
                              num_workers=1)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=1)
    trainer.fit(model, train_loader, val_loader)
    
    test_ds = BaseTestDataset(tokenizer=tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, drop_last=False, pin_memory=False,
                              num_workers=4)
    trainer.test(model,test_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--nepochs', type=int, default=1)
    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--full_data', action='store_true')
    parser.add_argument('--val_size', type=float, default=0.1)

    args = parser.parse_args()
    config = ""
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
