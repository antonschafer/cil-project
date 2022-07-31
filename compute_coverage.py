import numpy as np
import os
from utils import load_wandb_file
import argparse


def compute_coverage(runs, args):
    data = {}

    for j, t in enumerate(runs):
        data[t] = [np.load(load_wandb_file("val_correct_preds.npy", t, args.save_dir))]

    for i, (k, v) in enumerate(data.items()):
        print(k + " : " + str(np.array(v).mean()))
        for j, (k1, v1) in enumerate(data.items()):
            if j <= i:
                continue
            print(k + " + " + k1 + " : " + str((np.array(v) | np.array(v1)).mean()))
            for t, (k2, v2) in enumerate(data.items()):
                if t <= j:
                    continue
                print(
                    k
                    + " + "
                    + k1
                    + " + "
                    + k2
                    + " : "
                    + str((np.array(v) | np.array(v1) | np.array(v2)).mean())
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_runs", type=str, default=[], nargs="*")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join("/cluster/scratch", os.environ["USER"]),
    )

    args = parser.parse_args()
    config = vars(args)
    compute_coverage(args.model_runs, args)
