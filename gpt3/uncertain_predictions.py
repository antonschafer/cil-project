import argparse
import os
from utils import load_wandb_file, get_base_datasets, MODELS
import numpy as np
import matplotlib.pyplot as plt

def plot_uncertain_predictions(preds, labels, title, save_path):
    bin_preds = preds > 0.5
    mis_preds = bin_preds != labels

    def uncertain(thresh):
        return np.abs(preds - 0.5) <= thresh

    def get_n_mis_pred_uncertain(thresh):
        unc = uncertain(thresh)
        return np.sum(mis_preds & unc)

    total_mis_preds = sum(mis_preds)
    total_preds = len(preds)

    x = np.concatenate([np.linspace(0, 0.4, 101), np.linspace(0.4, 0.45, 101), np.linspace(0.45, 0.5, 501)])
    frac_mispred = [get_n_mis_pred_uncertain(t)/total_mis_preds for t in x]
    frac_total = [uncertain(t).sum()/total_preds for t in x]


    # create 2 plots next to each other
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # plot fraction mispredicted and fraction total vs x in left plot
    axs[0].plot(x, frac_mispred, label="fraction mispredicted")
    axs[0].plot(x, frac_total, label="fraction total")
    axs[0].set_xlabel("distance from 0.5 <= ...")
    axs[0].legend()

    axs[1].plot(frac_total, frac_mispred, label="tradeoff")
    axs[1].set_xlabel("fraction total")
    axs[1].set_ylabel("fraction mipredicted")
    axs[1].legend()

    fig.suptitle(title)
    plt.savefig(save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
        default=os.path.join("/cluster/scratch", os.environ["USER"]))
    parser.add_argument('--model', type=str)
    parser.add_argument('--run_id', type=str)
    args = parser.parse_args()

    tokenizer_name = MODELS[args.model]["tokenizer_name"]
                    

    config = dict(full_data=True, save_dir=args.save_dir, model=args.model,
        tokenizer_name=tokenizer_name)

    datasets = get_base_datasets(config)

    labels = datasets[1].labels[:,1].bool().numpy()
    preds = np.load(load_wandb_file("val_preds.npy", args.run_id, args.save_dir))
    assert labels.shape == preds.shape

    plot_uncertain_predictions(preds, labels,
        "Validation Mispredictions by Prediction Uncertainty",
        os.path.join("./gpt3/mispred_plots/{}.png".format(args.run_id)))





