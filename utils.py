# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:40:23 2018

@author: Clemens Biehl, Fabian Otto, Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import os
import glob
import pickle
import numpy as np
import pandas as pd

from random import shuffle

# -----------------------------------------------------------------------------
# Load data
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
            dtype={
                0: str,
                1: str
            })
        list_.append(df)
        
    frame = pd.concat(list_, axis=0, ignore_index=True)
    mask = pd.isna(frame[0])
    data = split(frame, mask)
    save_data_list(data)
    
    print("Finished loading data.")
    return data


def split(df, mask):
    """
    Splits data frame into sentences.
    
    :param df:          data frame to split
    :param mask:        mask indicating where to split
    """
    result = []
    start = 0
    
    for i in range(df.shape[0] - 1):
        if mask[i]:
            # get tuples from subset
            result.append([tuple(x) for x in df[start:i].values])
            start = i + 1
    
    return result


def save_data_list(data_list):
    """
    Saves data list to a text file.
    
    :param data_list:   data list to be saved
    """
    with open("data.txt", "wb") as f:
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