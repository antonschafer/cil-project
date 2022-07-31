import argparse
import os

from datasets.base_dataset import BaseDataset
from datasets.base_testdataset import BaseTestDataset
from transformers import AutoTokenizer
from utils import load_pickle, load_wandb_file, get_base_datasets, MODELS, write_pickle
import numpy as np
import matplotlib.pyplot as plt
COST_PER_TOKEN = 0.012 / 1000
UNC_THRESHS = np.concatenate([np.linspace(0, 0.4, 101), np.linspace(0.4, 0.45, 101), np.linspace(0.45, 0.5, 501)])




def uncertain(preds, thresh):
    return np.abs(preds - 0.5) <= thresh

def n_most_uncertain(preds, correct_preds, n):
    indexes = (np.abs(preds - 0.5) + correct_preds).argsort()[:n]
    final = np.full_like(preds, False, dtype=bool)
    final[indexes] = True
    return final

def get_n_mis_pred_uncertain(preds, labels, thresh):
    unc = uncertain(preds, thresh)
    mis_preds = (preds > 0.5) != labels
    return np.sum(mis_preds & unc)


def plot_uncertain_predictions(preds, labels, title, save_path):
    total_mis_preds = sum((preds > 0.5) != labels)
    total_preds = len(preds)

    x = UNC_THRESHS
    frac_mispred = [get_n_mis_pred_uncertain(preds, labels, t) / total_mis_preds for t in x]
    frac_total = [uncertain(preds, t).sum() / total_preds for t in x]

    plt.clf()
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
    print("Saved plot to", save_path)


def get_and_plot_uncertain_predictions(datasets, preds, cost_lim, path_plot, path_masks):
    total_costs = []
    perc_mispreds = []
    perc_test = []
    perc_valfinal = []

    labels = datasets["val_final"].labels
    total_mispreds = sum((preds["val_final"] > 0.5) != labels)

    tokens_test = np.array([len(x) for x in datasets["test"].data])
    tokens_valfinal = np.array([len(x) for x in datasets["val_final"].data])

    masks = None
    mispred_coverage = None
    total_coverage = None

    for t in UNC_THRESHS:
        unc_test = uncertain(preds["test"], t)
        cost_test = tokens_test[unc_test].sum() * COST_PER_TOKEN

        unc_valfinal = uncertain(preds["val_final"], t)
        n_mispred_valfinal = get_n_mis_pred_uncertain(preds["val_final"], labels, t)
        cost_valfinal = tokens_valfinal[unc_valfinal].sum() * COST_PER_TOKEN

        curr_mispred_coverage = n_mispred_valfinal / total_mispreds
        perc_mispreds.append(curr_mispred_coverage)
        perc_test.append(unc_test.sum() / len(preds["test"]))
        curr_total_coverage = unc_valfinal.sum() / len(preds["val_final"])
        perc_valfinal.append(curr_total_coverage)

        cost_sum = cost_test + cost_valfinal
        total_costs.append(cost_sum)

        if cost_sum <= cost_lim:
            masks = unc_test, unc_valfinal
            mispred_coverage = curr_mispred_coverage
            total_coverage = curr_total_coverage

    print("\nCan cover {:.2f}% of mispredictions ({:.2f}% of all) with cost {}$\n".format(mispred_coverage * 100,
                                                                                          total_coverage * 100,
                                                                                          cost_lim))

    plt.clf()
    plt.plot(total_costs, perc_mispreds, label="fraction of mispredictions on val_final set covered")
    plt.plot(total_costs, perc_valfinal, label="fraction of val_final set covered")
    plt.plot(total_costs, perc_test, label="fraction of test set covered")
    plt.xlabel("cost of inference on test and val_final")
    plt.legend()
    plt.title("Coverage by cost")
    plt.grid(True, "both")
    plt.savefig(path_plot)
    print("Saved plot to", path_plot)

    write_pickle(masks, path_masks)
    print("Saved masks to", path_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--cost_lim', type=float)
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join("/cluster/scratch", os.environ["USER"]))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    datasets, preds = {}, {}
    for split in ["val_final", "test"]:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if split == "test":
            dataset = BaseTestDataset(tokenizer=tokenizer, transform=None, pad=False)
        else:
            dataset = BaseDataset(split=split, tokenizer=tokenizer, full_data=True, transform=None, pad=False)
            dataset.labels = dataset.labels[:, 1].bool().numpy()
        datasets[split] = dataset
        preds[split] = np.load(load_wandb_file("{}_preds.npy".format(split), args.run_id, args.save_dir))#


    plot_uncertain_predictions(preds["val_final"], datasets["val_final"].labels,
                               "Val Final Mispredictions by Prediction Uncertainty",
                               "../gpt3/mispred_plots/{}_ratios.png".format(args.run_id))

    get_and_plot_uncertain_predictions(
        datasets,
        preds,
        args.cost_lim,
       "../gpt3/mispred_plots/{}_coverage_cost.png".format(args.run_id),
        "../gpt3/masks/{}_{}.pkl".format(args.run_id, str(args.cost_lim).replace(".", "-")),
    )
