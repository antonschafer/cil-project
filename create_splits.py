import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
from utils import DATA_VERSION


random.seed(0)
np.random.seed(0)

def read_txt(filename):
    with open(filename) as f:
        data = f.read().split("\n")
        data = [x for x in data if x != ""]
    return set(data)

def split_csv(full, val_size, val_final_size):

    suffix = "_full" if full else ""

    data_neg = list(read_txt('./twitter-datasets/train_neg{}.txt'.format(suffix)))
    data_pos = list(read_txt('./twitter-datasets/train_pos{}.txt'.format(suffix)))

    all_x = data_pos + data_neg
    all_y = [1 for _ in range(len(data_pos))] + [0 for _ in range(len(data_neg))]
    all_data = {'texts':all_x,'labels':all_y}
    df = pd.DataFrame(all_data)
    df = shuffle(df)
    print("Total number of tweets:", len(df))
    print("Overlap between pos and neg:", len(set(data_pos).intersection(data_neg)))

    train_size = df.shape[0] - (val_final_size + val_size)
    print("Train, val, val final sizes",train_size, val_size, val_final_size)
    df_train = df[: train_size]
    df_val = df[train_size: train_size+val_size]
    df_val_final = df[train_size+val_size:]

    prefix = "full" if full else "small"
    df_train.to_csv("./twitter-datasets/{}_train_v{}.csv".format(prefix, DATA_VERSION),index=False)
    df_val.to_csv("./twitter-datasets/{}_val_v{}.csv".format(prefix, DATA_VERSION),index=False)
    df_val_final.to_csv("./twitter-datasets/{}_val_final_v{}.csv".format(prefix, DATA_VERSION),index=False)

    return


if __name__ == '__main__':
    print("Generating full datasplit")
    split_csv(full=True, val_size=100000, val_final_size=50000)
    print("\nGenerating small datasplit")
    split_csv(full=False, val_size=10000, val_final_size=5000)
