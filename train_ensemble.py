from torch.utils.data import DataLoader
import wandb
from models.binary_hf_module import BinaryHFModule
from models.ensemble_module import EnsembleModule
from models.three_class_hf_module import ThreeClassHFModule

from utils import get_base_arg_parser, get_base_datasets, get_trainer


def train(config):
    # TODO add submodels and in_dim
    model = EnsembleModule(config=config, submodels=None, in_dim=None)
    trainer = get_trainer(config)
    _, val_set, val_final_set, test_set = get_base_datasets(config)

    train_loader = DataLoader(
        val_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=1)
    val_loader = DataLoader(
        val_final_set, batch_size=config['batch_size'], num_workers=1)
    test_loader = DataLoader(
        test_set, batch_size=config['batch_size'], num_workers=4)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == '__main__':

    parser = get_base_arg_parser()
    args = parser.parse_args()
    config = vars(args)
    print('Config:')
    print(config)
    train(config)
