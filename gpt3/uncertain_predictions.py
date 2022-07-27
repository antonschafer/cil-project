import argparse
import os
from datasets.version import DATA_VERSION
import openai
from transformers import AutoTokenizer
from datasets.base_dataset import BaseDataset
from datasets.base_testdataset import BaseTestDataset
from gpt3.prepare_data import save_dir
from utils import load_pickle, load_wandb_file, get_base_datasets, MODELS, write_pickle
import numpy as np
from utils import get_base_arg_parser, get_base_datasets, get_bert_config, get_trainer
import matplotlib.pyplot as plt
import pandas as pd
import time
COST_PER_TOKEN = 0.012 / 1000
UNC_THRESHS = np.concatenate([np.linspace(0, 0.4, 101), np.linspace(0.4, 0.45, 101), np.linspace(0.45, 0.5, 501)])


def get_uncertain_from_run(run, save_dir):
    preds = np.load(load_wandb_file('train_ensemble_preds.npy', run, save_dir))
    correct_preds = np.load(load_wandb_file('train_ensemble_correct_preds.npy', run, save_dir))

    uncertain_keep = n_most_uncertain(preds, correct_preds, 1000)

    random_train = np.random.choice(np.where(~uncertain_keep)[0], 1000, replace=False)
    train = np.append(random_train, np.where(uncertain_keep))

    filename = "../twitter-datasets/full_train_ensemble" +  "_v{}.csv".format(
        DATA_VERSION)
    df = pd.read_csv(filename)
    tweets = df.iloc[train]
    tweets = tweets.rename(columns={'texts': 'prompt', 'labels': 'completion'})

    promptt = "Tweet: {}\n Sentiment: "
    labels = {
        1: "positive",
        0: "negative"
    }
    tweets["prompt"] = tweets["prompt"].apply(lambda x: promptt.format(x))
    tweets["completion"] = tweets["completion"].apply(lambda x: labels[x])
    tweets.to_csv('openai-parsed.csv', index=False)

def get_final_uncertain_from_run(run, save_dir):
    val_final = np.load(load_wandb_file('val_final_preds.npy', run, save_dir))
    val_final_preds = val_final >= .5
    filename = "../twitter-datasets/full_val_final" +  "_v{}.csv".format(
        DATA_VERSION)
    df = pd.read_csv(filename)
    correct_labels = df.labels
    masks = np.load('../gpt3/masks/1vox48hx_4-0.pkl', allow_pickle=True)
    val_final_mask = masks[1]
    to_pred = df[val_final_mask]
    promptt = "{}\n Sentiment: "
    tweets = to_pred.texts.apply(lambda x: promptt.format(x))
    openai.api_key = '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'

    preds = [False] * len(tweets)
    for j, i in enumerate(tweets):
        w = openai.Completion.create(model="curie:ft-personal-2022-07-27-15-32-23", prompt=i, temperature=0,
                                     max_tokens=1)
        preds[j] = w['choices'][0]['text'] == ' positive'

    val_final_preds[tweets.index] = preds
    pd.DataFrame(val_final_preds).to_csv(f'final_val_gpt3_{run}.csv')
    print((val_final_preds == correct_labels).mean())

    pass

def load_data():
    def read_txt(filename):
        with open(filename) as f:
            data = f.read().split("\n")
            data = [x for x in data if x != ""]
        # drop indices
        return [",".join(x.split(",")[1:]) for x in data]

    return read_txt('../twitter-datasets/test_data.txt')

def get_test_uncertain_from_run(run, save_dir):
    test = np.load(load_wandb_file('test_preds.npy', run, save_dir))
    test_preds = test >= .5
    df = pd.DataFrame(load_data(), columns=['texts'])
    masks = np.load('../gpt3/masks/1vox48hx_4-0.pkl', allow_pickle=True)
    test_mask = masks[0]
    to_pred = df[test_mask]
    promptt = "{}\n Sentiment: "
    tweets = to_pred.texts.apply(lambda x: promptt.format(x))
    openai.api_key = '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'

    preds = [False] * len(tweets)
    for j, i in enumerate(tweets):
        w = openai.Completion.create(model="curie:ft-personal-2022-07-27-15-32-23", prompt=i, temperature=0,
                                     max_tokens=1)
        preds[j] = w['choices'][0]['text'] == ' positive'
        time.sleep(2)

    test_preds[tweets.index] = preds
    final = pd.DataFrame(test_preds, columns=['Prediction']).reset_index(inplace=False).rename(columns = {'index':'Id'}, inplace = False)
    final.Prediction = final.Prediction.replace({True: 1, False: 0})
    final.to_csv(f'test{run}.csv', index=False)

    pass




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

    #get_uncertain_from_run('1vox48hx', args.save_dir)
    get_test_uncertain_from_run('1vox48hx', args.save_dir)

    #datasets, preds = {}, {}
    #for split in ["val_final", "test"]:
    ##    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #    if split == "test":
    #        dataset = BaseTestDataset(tokenizer=tokenizer, transform=None, pad=False)
    #    else:
    #        dataset = BaseDataset(split=split, tokenizer=tokenizer, full_data=True, transform=None, pad=False)
    #        dataset.labels = dataset.labels[:, 1].bool().numpy()
    #    datasets[split] = dataset
    #    preds[split] = np.load(load_wandb_file("{}_preds.npy".format(split), args.run_id, args.save_dir))#


    #plot_uncertain_predictions(preds["val_final"], datasets["val_final"].labels,
    #                           "Val Final Mispredictions by Prediction Uncertainty",
    #                           "../gpt3/mispred_plots/{}_ratios.png".format(args.run_id))

    #get_and_plot_uncertain_predictions(
    #    datasets,
    #    preds,
    ##    args.cost_lim,
     #   "../gpt3/mispred_plots/{}_coverage_cost.png".format(args.run_id),
    #    "../gpt3/masks/{}_{}.pkl".format(args.run_id, str(args.cost_lim).replace(".", "-")),
    #)
