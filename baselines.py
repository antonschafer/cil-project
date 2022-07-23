import argparse
import os
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
import re
import wandb
import random
random.seed(0)
np.random.seed(0)
import torchtext
import torch
from IPython import embed

def load_data(full_data):

    def read_txt(filename):
        with open(filename) as f:
            data = f.readlines()
        return set(data)


    train_filename = "./twitter-datasets/" + ("full_" if full_data else "small_") + "train.csv"
    val_filename = "./twitter-datasets/" + ("full_" if full_data else "small_") + "val.csv"
    valfinal_filename = "./twitter-datasets/" + ("full_" if full_data else "small_") + "val_final.csv"

    train_df = pd.read_csv(train_filename)
    val_df = pd.read_csv(val_filename)
    valfinal_df = pd.read_csv(valfinal_filename)
    
    test_data = read_txt('./twitter-datasets/test_data.txt')

    return train_df, val_df, valfinal_df, test_data

def clean_texts(texts,stopwords):

    clean_texts = []
    for text in texts:
        clean_text = re.sub('[^a-zA-Z]', ' ', text)
        clean_text = clean_text.lower()
        clean_text = clean_text.split()
        #clean_text = [lemmatizer.lemmatize(word) for word in clean_text if not word in set(stopwords)]
        clean_text = ' '.join(clean_text)
        clean_texts.append(clean_text)

    return clean_texts

def preds_metrics(preds,labels):

    print(preds.shape)
    print(labels.shape)
    print(classification_report(labels, preds))
    accuracy = np.mean(preds == labels)
    print(accuracy)
    return accuracy


def texts_to_glove(texts,glove_embedder):
    text_gloves = []
    for text in texts:
        glove_text = tuple(glove_embedder[w] for w in text)
        #text_gloves.append(torch.vstack(glove_text).sum(axis=0).numpy())
        text_gloves.append(torch.vstack(glove_text).float().numpy())
    return (text_gloves)

def run_glove(config):

    wandb.init(project="glove")

    train_data, val_data, val_final_data, test_data = load_data(config["full_data"])
    stopwords = nltk.corpus.stopwords.words('english')


    train_data["texts_clean"] = clean_texts(train_data["texts"], stopwords)
    val_data["texts_clean"] = clean_texts(val_data["texts"], stopwords)
    val_final_data["texts_clean"] = clean_texts(val_final_data["texts"], stopwords)
    test_data_clean = clean_texts(test_data, stopwords)
    
    glove = torchtext.vocab.GloVe(name="6B", dim=100, max_vectors=20000)

    print("Getting glove vectors")
    train_data_glove = texts_to_glove(train_data["texts_clean"],glove)
    val_data_glove = texts_to_glove(val_data["texts_clean"],glove)
    val_final_data_glove = texts_to_glove(val_final_data["texts_clean"],glove)
    test_data_glove = texts_to_glove(test_data_clean,glove)

    maxlen = 104
    train_data_glove = tf.keras.preprocessing.sequence.pad_sequences(train_data_glove, maxlen=maxlen)
    val_data_glove = tf.keras.preprocessing.sequence.pad_sequences(val_data_glove, maxlen=maxlen)
    val_final_data_glove = tf.keras.preprocessing.sequence.pad_sequences(val_final_data_glove, maxlen=maxlen)
    test_data_glove = tf.keras.preprocessing.sequence.pad_sequences(test_data_glove, maxlen=maxlen)

    model = tf.keras.Sequential()

    model.add(Bidirectional(LSTM(100, return_sequences=True),
                            input_shape=(104, 100)))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))

    model.add(Bidirectional(LSTM(10,return_sequences=False)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    opti = tf.keras.optimizers.Adam(
        learning_rate=0.001,
    )

    def scheduler(epoch, lr):
        return lr * tf.math.exp(-0.1)
    
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


    model.compile(loss='binary_crossentropy', optimizer=opti)

    print(train_data_glove.shape)
    model.fit(train_data_glove, train_data["labels"], batch_size=128, epochs=5,callbacks=[callback], validation_data=(val_data_glove,val_data["labels"]),verbose=2)

    val_preds = model.predict(val_data_glove)
    
    val_preds[val_preds>0.5] = 1
    val_preds[val_preds<=0.5] = 0
    val_preds = np.squeeze(val_preds)
    val_acc = preds_metrics(val_preds,val_data["labels"])

    valfinal_preds = model.predict(val_final_data_glove)
    valfinal_preds[valfinal_preds>0.5] = 1
    valfinal_preds[valfinal_preds<=0.5] = 0
    valfinal_preds = np.squeeze(valfinal_preds)
    val_final_acc = preds_metrics(valfinal_preds,val_final_data["labels"])

    wandb.log({"val_acc": val_acc,"val_final_acc": val_final_acc})
    wandb.summary["val_acc"] = val_acc
    wandb.summary["val_final_acc"] = val_final_acc
    
    test_outputs = model.predict(test_data_glove)
    test_outputs = np.squeeze(test_outputs)
    test_outputs[test_outputs>0.5] = 1
    test_outputs[test_outputs<=0.5] = 0
    test_outputs[test_outputs == 0] = -1
    ids = np.arange(1, test_outputs.shape[0]+1)
    outdf = pd.DataFrame({"Id": ids, 'Prediction': test_outputs})
    outdf.to_csv(os.path.join(config["save_dir"], 'glove_outputs.csv'), index=False)


def run_tfidf(config):

    wandb.init(project="tfidf")

    train_data, val_data, val_final_data, test_data = load_data(config["full_data"])
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    nltk.download('wordnet')

    stopwords = nltk.corpus.stopwords.words('english')


    train_data["texts_clean"] = clean_texts(train_data["texts"], stopwords)
    val_data["texts_clean"] = clean_texts(val_data["texts"], stopwords)
    val_final_data["texts_clean"] = clean_texts(val_final_data["texts"], stopwords)
    test_data_clean = clean_texts(test_data, stopwords)

    tf_idf = TfidfVectorizer()
    train_data_tf = tf_idf.fit_transform(train_data["texts_clean"])
    val_data_tf = tf_idf.transform(val_data["texts_clean"])
    val_final_data_tf = tf_idf.transform(val_final_data["texts_clean"])
    test_data_tf = tf_idf.transform(test_data_clean)
    
    print(" Train texts in tf idf embedding",train_data_tf.shape)
    clf = LogisticRegression(random_state=0,max_iter=100000).fit(train_data_tf, train_data["labels"])
    
    train_preds = clf.predict(train_data_tf)
    train_acc = preds_metrics(train_preds,train_data["labels"])

    val_preds = clf.predict(val_data_tf)
    val_acc = preds_metrics(val_preds,val_data["labels"])

    valfinal_preds = clf.predict(val_final_data_tf)
    val_final_acc = preds_metrics(valfinal_preds,val_final_data["labels"])

    wandb.log({"train_acc": train_acc,"val_acc": val_acc,"val_final_acc": val_final_acc})
    wandb.summary["train_acc"] = train_acc
    wandb.summary["val_acc"] = val_acc
    wandb.summary["val_final_acc"] = val_final_acc

    test_outputs = clf.predict(test_data_tf)
    test_outputs[test_outputs == 0] = -1
    ids = np.arange(1, test_outputs.shape[0]+1)
    outdf = pd.DataFrame({"Id": ids, 'Prediction': test_outputs})

    outdf.to_csv(os.path.join(wandb.run.dir, 'tfidf_outputs.csv'), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--full_data', action='store_true')
    
    parser.add_argument('--tfidf', action='store_true')
    parser.add_argument('--glove', action='store_true')

    parser.add_argument('--save_dir', type=str,default="")

    args = parser.parse_args()

    config = vars(args)

    if config["save_dir"] == "":
        config["save_dir"] = os.path.join("/cluster/scratch", os.environ["USER"])

    if args.tfidf:
        run_tfidf(config)

    if args.glove:
        run_glove(config)


