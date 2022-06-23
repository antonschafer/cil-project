import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from utils import get_base_datasets, get_bert_config


def train(module, config):
    model = module(config=config)

    callbacks = [EarlyStopping(monitor="val_loss", mode="min"),
                 ModelCheckpoint(monitor='val_loss', dirpath=config['save_path'], filename="model.ckpt")]
    trainer = pl.Trainer(max_epochs=config['nepochs'], gpus=config["gpus"], callbacks=callbacks,
                         check_val_every_n_epoch=config['val_freq'], gradient_clip_val=1)

    train_set, val_set, test_set = get_base_datasets(config)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=False,
                              num_workers=1)
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, drop_last=False, pin_memory=False,
                             num_workers=4)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--nepochs', type=int, default=1)
    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer_name', type=str,
                        default='')  # default same as model_name

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--full_data', action='store_true')

    args = parser.parse_args()
    config, module = get_bert_config(args)

    train(config, module)
