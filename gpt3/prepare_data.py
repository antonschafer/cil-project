import argparse
import pandas as pd


prompt = "Twitter Sentiment Analysis Examples.\n\n Tweet: {}\n Sentiment: "
labels = {1: "positive", 0: "negative"}

save_dir = "./gpt3/input_data/"


def load_data(split):
    filename = "./twitter-datasets/full_" + split + ".csv"
    return pd.read_csv(filename)


def split_data(df, frac, seed):
    return df.sample(frac=frac, random_state=seed)


def prepare(df, is_test=False):
    cols = {}
    cols["prompt"] = df["texts"].apply(lambda x: prompt.format(x))
    if not is_test:
        cols["completion"] = df["labels"].apply(lambda x: labels[x])
    return pd.DataFrame(cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_frac", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # train
    df_train = load_data("train")
    df_train = split_data(df_train, args.train_frac, args.seed)
    df_train = prepare(df_train)
    df_train.to_csv(
        save_dir
        + "train_{}_{}.csv".format(str(args.train_frac).replace(".", "-"), args.seed)
    )

    # val
    df_val = load_data("val")
    df_val = prepare(df_val)
    df_val.to_csv(save_dir + "val.csv")

    # val_final
    df_val_final = load_data("val_final")
    df_val_final = prepare(df_val_final)
    df_val_final.to_csv(save_dir + "val_final.csv")

    # test
    with open("twitter-datasets/test_data.txt") as f:
        df_test = pd.DataFrame({"texts": list(f.readlines())})
    df_test = prepare(df_test, is_test=True)
    df_test.to_csv(save_dir + "test.csv")
