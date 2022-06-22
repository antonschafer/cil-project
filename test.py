import argparse
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modeling import *
from modeling.twitter_roberta.twitter_roberta_module import TwitterRobertaModule
from utils import get_base_datasets, get_bert_config

MODULES = {
    "base": BaseModule,
    "twitter_roberta": TwitterRobertaModule
}


def test(config):
    model = MODULES[config["module"]](config=config)
    if config["save_path"] != "":
        model.load_ckpt(config['save_path'])
    trainer = pl.Trainer(gpus=config["gpus"])

    _, val_set, test_set = get_base_datasets(config)

    if config["validation"]:
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False, drop_last=False, pin_memory=True,
                                  num_workers=4)
        trainer.validate(model, val_loader)
    else:
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, pin_memory=True,
                                  num_workers=4)
        trainer.test(model, test_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--module', type=str, default='base')
    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer_name', type=str, default='')  # default same as model
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--full_data', action='store_true', help="use full validation data")

    args = parser.parse_args()
    config = get_bert_config(args)

    test(config)
