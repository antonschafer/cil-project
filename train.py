import argparse
from torch.utils.data import DataLoader

from utils import get_base_arg_parser, get_base_datasets, get_bert_config, compute_metrics, get_trainer


def train(config, module):
    model = module(config=config)

    trainer = get_trainer(config)

    train_set, val_set, _, test_set = get_base_datasets(config)

    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=1)
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], num_workers=1)
    test_loader = DataLoader(
        test_set, batch_size=config['batch_size'], num_workers=4)

    # TODO also run on val_final set (make sure to log with metrics with proper name, not just val_acc)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    try:
        compute_metrics(
            model, val_set, config['batch_size'], config["run_name"])
    except:
        pass


if __name__ == '__main__':

    parser = get_base_arg_parser()
    parser.add_argument('--model', type=str, default='base')

    args = parser.parse_args()
    config, module = get_bert_config(args)

    train(config, module)
