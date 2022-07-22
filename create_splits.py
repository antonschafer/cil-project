import argparse
import os

import string
import pandas as pd
import pickle
import re
import logging
import traceback
import numpy as np
import random
from sklearn.utils import shuffle


random.seed(0)
np.random.seed(0)

def read_txt(filename):
    with open(filename) as f:
        data = f.read().split("\n")
        data = [x for x in data if x != ""]
    return set(data)

def split_csv_small():


    data_neg = list(read_txt('./twitter-datasets/train_neg.txt'))
    data_pos = list(read_txt('./twitter-datasets/train_pos.txt'))

    all_x = data_pos + data_neg
    all_y = [1 for _ in range(len(data_pos))] + [0 for _ in range(len(data_neg))]
    all_data = {'texts':all_x,'labels':all_y}
    df = pd.DataFrame(all_data)
    df = shuffle(df)
    print("All data shape",df.shape)
    #Size of all the data is 
    val_size = 10000
    val_final_size = 5000
    train_size = df.shape[0] - (val_final_size + val_size)
    
    print("Train, val, val final sizes",train_size, val_size, val_final_size)
    df_train = df[: train_size]
    df_val = df[train_size: train_size+val_size]
    df_val_final = df[train_size+val_size:]

    df_train = df_train[ ~ df_train["texts"].isin(df_val["texts"])]
    df_train = df_train[ ~ df_train["texts"].isin(df_val_final["texts"])]
    df_val = df_val[ ~ df_val["texts"].isin(df_val_final["texts"])]

    print("Train shape",df_train.shape)
    print("Val shape",df_val.shape)
    print("val final shape",df_val_final.shape)

    df_train.to_csv("./twitter-datasets/small_train.csv",index=False)
    df_val.to_csv("./twitter-datasets/small_val.csv",index=False)
    df_val_final.to_csv("./twitter-datasets/small_val_final.csv",index=False)

    return


def split_csv_full():


    data_neg = list(read_txt('./twitter-datasets/train_neg_full.txt'))
    data_pos = list(read_txt('./twitter-datasets/train_pos_full.txt'))

    all_x = data_pos + data_neg
    all_y = [1 for _ in range(len(data_pos))] + [0 for _ in range(len(data_neg))]
    all_data = {'texts':all_x,'labels':all_y}
    df = pd.DataFrame(all_data)
    df = shuffle(df)
    print("All data shape",df.shape)
    #Size of all the data is 
    val_size = 100000
    val_final_size = 50000
    train_size = df.shape[0] - (val_final_size + val_size)
    print("Train, val, val final sizes",train_size, val_size, val_final_size)
    df_train = df[: train_size]
    df_val = df[train_size: train_size+val_size]
    df_val_final = df[train_size+val_size:]

    df_train = df_train[ ~ df_train["texts"].isin(df_val["texts"])]
    df_train = df_train[ ~ df_train["texts"].isin(df_val_final["texts"])]
    df_val = df_val[ ~ df_val["texts"].isin(df_val_final["texts"])]
    print("Train shape",df_train.shape)
    print("Val shape",df_val.shape)
    print("val final shape",df_val_final.shape)
    df_train.to_csv("./twitter-datasets/full_train.csv",index=False)
    df_val.to_csv("./twitter-datasets/full_val.csv",index=False)
    df_val_final.to_csv("./twitter-datasets/full_val_final.csv",index=False)

    return


if __name__ == '__main__':

    split_csv_full()       

    split_csv_small()                                                                                                          
