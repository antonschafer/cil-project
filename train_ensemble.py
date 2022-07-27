import warnings
from torch.utils.data import DataLoader
from datasets.ensemble_dataset import EnsembleDataset
from models.ensemble_module import EnsembleModule
from test import run_eval

from utils import get_base_arg_parser, get_trainer


def train(config):
    train_ensemble_set = EnsembleDataset(
        runs=config["model_runs"], split="train_ensemble", save_dir=config["save_dir"])
    val_set = EnsembleDataset(
        runs=config["model_runs"], split="val", save_dir=config["save_dir"])
    val_final_set = EnsembleDataset(
        runs=config["model_runs"], split="val_final", save_dir=config["save_dir"])
    test_set = EnsembleDataset(
        runs=config["model_runs"], split="test", save_dir=config["save_dir"])

    model = EnsembleModule(config=config, in_dim=val_set.dim)
    trainer = get_trainer(config)

    train_ensemble_loader = DataLoader(
        train_ensemble_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=1)
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], num_workers=1)

    trainer.fit(model, train_ensemble_loader, val_dataloaders=val_loader)
    run_eval(model, ckpt_path=trainer.checkpoint_callback.best_model_path,
             train_ensemble_set=train_ensemble_set, val_set=val_set, val_final_set=val_final_set, test_set=test_set)


if __name__ == '__main__':

    parser = get_base_arg_parser()
    parser.add_argument("--model_runs", type=str,
                        default=[], nargs="*")  # TODO default
    parser.add_argument("--dropout", type=float,
                        default=0.2)
    parser.add_argument("--hidden_size", type=int,
                        default=512)

    args = parser.parse_args()
    config = vars(args)
    print('Config:')
    print(config)

    # we are using 1 worker and that's ok
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    # we don't want to be warned that awndb run dir where checkpoint is saved is not empty
    warnings.filterwarnings("ignore", ".*exists and is not empty*")

    train(config)
