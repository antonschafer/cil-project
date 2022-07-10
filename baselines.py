import argparse
import os
from utils import get_base_datasets, get_bert_config, compute_metrics
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_data(full_data):

    def read_txt(filename):
        with open(filename) as f:
            data = f.readlines()
        return set(data)

    if full_data:
        data_neg = read_txt('./twitter-datasets/train_neg_full.txt')
        data_pos = read_txt('./twitter-datasets/train_pos_full.txt')
    else:
        data_neg = read_txt('./twitter-datasets/train_neg.txt')
        data_pos = read_txt('./twitter-datasets/train_pos.txt')
    
    test_data = read_txt('./twitter-datasets/test_data.txt')
    return data_pos, data_neg

def clean_texts(texts,lemmatizer,stopwords):

    clean_texts = []
    for text in texts:
        clean_text = re.sub('[^a-zA-Z]', ' ', text)
        clean_text = clean_text.lower()
        clean_text = clean_text.split()
        clean_text = [lemmatizer.lemmatize(word) for word in clean_text if not word in set(stopwords)]
        clean_text = ' '.join(clean_text)
        clean_texts.append(clean_text)

    return clean_texts

def run_tfidf(config):

    #data_pos, data_neg,test_data = load_data(config["full_data"])
    train_data, val_data, val_final_data, test_data = get_base_datasets(config)
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    train_data_clean = clean_texts(train_data, lemmatizer, stopwords)
    val_data_clean = clean_texts(val_data, lemmatizer, stopwords)
    val_final_data_clean = clean_texts(val_final_data, lemmatizer, stopwords)
    test_data_clean = clean_texts(test_data, lemmatizer, stopwords)

    tf_idf = TfidfVectorizer()
    train_data_tf = tf_idf.fit_transform(train_data_clean)
    val_data_tf = tf_idf.fit_transform(val_data_clean)
    val_final_data_tf = tf_idf.fit_transform(val_final_data_clean)
    test_x_tf = tf_idf.transform(test_x_clean)
    
    print(" All texts in tf idf embedding",all_x_tf.shape)
    clf = LogisticRegression(random_state=0).fit(train_data_tf, y)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='')  # TODO needed?
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join("/cluster/scratch", os.environ["USER"]))


    parser.add_argument('--full_data', action='store_true')
    
    parser.add_argument('--tfidf', action='store_true')

    args = parser.parse_args()

    config, _ = get_bert_config(args)

    if args.tfidf:
        run_tfidf(config)


