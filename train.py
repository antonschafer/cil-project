import argparse
from torch.utils.data import DataLoader

from utils import get_base_arg_parser, get_base_datasets, get_bert_config, compute_metrics, get_trainer
from test import run_eval


def train(config, module):
    model = module(config=config)

    trainer = get_trainer(config)

    train_set, val_set, val_final_set, test_set = get_base_datasets(config)

    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=1)
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], num_workers=1)

    trainer.fit(model, train_loader, val_loader)

    run_eval(model, ckpt_path=trainer.checkpoint_callback.best_model_path,
             val_set=val_set, val_final_set=val_final_set, test_set=test_set)

#   TODO change s.t. can use logged predictions
#   try:
#         compute_metrics(
#             model, val_set, config['batch_size'], config["run_name"])
#     except:
#         pass
#


if __name__ == '__main__':

    parser = get_base_arg_parser()
    parser.add_argument('--model', type=str, default='base')

    args = parser.parse_args()
    config, module = get_bert_config(args)

    train(config, module)
