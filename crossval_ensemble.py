import warnings
from torch.utils.data import DataLoader
from datasets.ensemble_dataset import EnsembleDataset
from models.ensemble_module import EnsembleModule
from test import run_eval
from utils import get_base_arg_parser, get_trainer
import torch
import math


def cross_val(config):

    val_set = EnsembleDataset(
        runs=config["model_runs"], split="val", save_dir=config["save_dir"]
    )
    val_final_set = EnsembleDataset(
        runs=config["model_runs"], split="val_final", save_dir=config["save_dir"]
    )
    test_set = EnsembleDataset(
        runs=config["model_runs"], split="test", save_dir=config["save_dir"]
    )

    val_final_loader = DataLoader(
        val_final_set, batch_size=config["batch_size"], num_workers=1
    )

    indices = list(range(len(val_set)))

    for i in range(config["crossval_num"]):
        indices_val_x = indices[
            math.floor(len(val_set) * (i / config["crossval_num"])) : math.ceil(
                len(val_set) * ((i + 1) / config["crossval_num"])
            )
        ]
        indices_train_x = [idx for idx in indices if idx not in indices_val_x]
        assert (len(indices_val_x) + len(indices_train_x)) == len(indices)
        train_x_set = torch.utils.data.Subset(val_set, indices_train_x)
        val_x_set = torch.utils.data.Subset(val_set, indices_val_x)

        train_x_loader = DataLoader(
            train_x_set,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )

        val_x_loader = DataLoader(
            val_x_set,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )

        model = EnsembleModule(config=config, in_dim=val_set.dim)
        trainer = get_trainer(config)
        trainer.fit(
            model, train_x_loader, val_dataloaders=(val_final_loader, val_x_loader)
        )
        run_eval(
            model,
            ckpt_path=trainer.checkpoint_callback.best_model_path,
            val_set=None,
            val_final_set=val_final_set,
            test_set=test_set,
        )


if __name__ == "__main__":

    parser = get_base_arg_parser()
    parser.add_argument("--model_runs", type=str, default=[], nargs="*")  # TODO default
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--crossval_num", type=int, default=5)

    args = parser.parse_args()
    config = vars(args)
    print("Config:")
    print(config)

    # we are using 1 worker and that's ok
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    # we don't want to be warned that awndb run dir where checkpoint is saved is not empty
    warnings.filterwarnings("ignore", ".*exists and is not empty*")

    cross_val(config)
