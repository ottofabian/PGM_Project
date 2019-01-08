# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:40:23 2018

@author: Clemens, Fabian Otto, Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import glob
import pickle
import itertools

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import shuffle


# -----------------------------------------------------------------------------
# Util functions
# -----------------------------------------------------------------------------

def preprocess_raw_data(path="./gmb-2.2.0/data", max_=None, load_entities=True):
    """
    Loads the data set.

    :param path:            path to the data root folder
    :param max_:            number of documents to be loaded
    :param load_entities:   flag indicating whether to load entities or not
    :return:                pandas data frame containing the documents
    """
    print("Loading data...")
    all_files = glob.glob(os.path.join(path, "*/*/en.tags"))

    list_ = []
    all_files = all_files[:max_]

    for file_ in all_files:
        print(file_)
        df = pd.read_csv(
            file_,
            index_col=None,
            usecols=[0, 1 if not load_entities else 3],
            header=None,
            sep="\t",
            skip_blank_lines=False,
            quotechar="\'",
            dtype={
                0: str,
                1: str
            })
        list_.append(df)

    frame = pd.concat(list_, axis=0, ignore_index=True)
    mask = pd.isna(frame[0])
    data = split(frame, mask)

    path = "data_POS.txt" if not load_entities else "data_NER.txt"
    save_data_list(data, path)

    print("Finished loading data.")
    return data


def split(df, mask):
    """
    Splits data frame into sentences.

    :param df:          data frame to split
    :param mask:        mask indicating where to split
    :return:            split data set
    """
    result = []
    start = 0

    for i in range(df.shape[0] - 1):
        if mask[i]:
            # get tuples from subset
            result.append([tuple(x) for x in df[start:i].values])
            start = i + 1

    return result


def save_data_list(data_list, path):
    """
    Saves data list to a text file.

    :param data_list:   data list to be saved
    :param path:        path to save the file to
    """
    with open(path, "wb") as f:
        pickle.dump(data_list, f)


def load_data_list(file_name):
    """
    Reads data list file.

    :param file_name:   name of the file to be loaded
    :return:            data list
    """
    with open(file_name, "rb") as f:
        data_list = pickle.load(f)

    return data_list


def train_test_split(data, train_ratio=0.8):
    """
    Splits the data into training and test set.

    :param data:        data to be slit into training and test set
    :param test_ratio:  percentage of test data
    :return:            data_train, data_test
    """
    # shuffle data before splitting
    shuffle(data)
    idx = int(len(data) * train_ratio)
    return data[:idx], data[idx:]


def separate_labels_from_features(X):
    """
    Separates labels from features.

    :param X:           words with labels in the form: [[(w1, t1), (w2, t2), ...], [...], ...]
    :return:            features, labels
    """
    features = []
    labels = []

    # iterate over all sentences
    for sent in X:
        features_sub = []
        labels_sub = []

        # iterate over all words in the sentence
        for (w, t) in sent:
            features_sub.append(w)
            labels_sub.append(t)

        # add sublist to overall result
        features.append(features_sub)
        labels.append(labels_sub)

    return features, labels


def show_misclassifications(gold_labels, pred_labels):
    """
    Show misclassifications made by the model.

    :param gold_labels: true labels
    :param pred_labels: predicted labels
    """
    # flatten lists for comparison
    flatten = lambda l: [item for sublist in l for item in sublist]
    gold_labels = flatten(gold_labels)
    pred_labels = flatten(pred_labels)

    for i in range(len(gold_labels)):
        if pred_labels[i] != gold_labels[i][1]:
            print(gold_labels[i][0], "\t", 
                  pred_labels[i], "\t", gold_labels[i][1])


def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    :param cm:          confusion matrix object
    :param classes:
    """
    np.set_printoptions(precision=2)
    plt.figure(figsize=(20, 20))
    matplotlib.rcParams.update({"font.size": 10})

    # normalize confusion matrix
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # plot confusion matrix
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], ".2f"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    plt.show()
