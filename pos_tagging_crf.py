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
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

def load_data(max_=None, load_entities=True):
    """
    Loads the data set.
    
    :param max_:            number of documents to be loaded
    :param load_entities:   flag indicating whether to load entities or not
    :return:                pandas data frame containing the documents
    """
    print("Loading data...")
    path = "./gmb-2.2.0/data"
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
            dtype={
                0: str,
                1: str
            })
        list_.append(df)
        
    frame = pd.concat(list_, axis=0, ignore_index=True)
    print("Finished loading data.")
    return frame


def extract_features(df):
    """
    Extracts the features from the data.
    
    :param df:      pandas data frame containing the data
    :return:        extracted features
    """
    pass


#def word2features(sent, i):
#    word = sent[i][0]
#    postag = sent[i][1]
#
#    features = {
#        'bias': 1.0,
#        'word.lower()': word.lower(),
#        'word[-3:]': word[-3:],
#        'word.isupper()': word.isupper(),
#        'word.istitle()': word.istitle(),
#        'word.isdigit()': word.isdigit(),
#        'postag': postag,
#        'postag[:2]': postag[:2],
#    }
#    if i > 0:
#        word1 = sent[i-1][0]
#        postag1 = sent[i-1][1]
#        features.update({
#            '-1:word.lower()': word1.lower(),
#            '-1:word.istitle()': word1.istitle(),
#            '-1:word.isupper()': word1.isupper(),
#            '-1:postag': postag1,
#            '-1:postag[:2]': postag1[:2],
#        })
#    else:
#        features['BOS'] = True
#
#    if i < len(sent)-1:
#        word1 = sent[i+1][0]
#        postag1 = sent[i+1][1]
#        features.update({
#            '+1:word.lower()': word1.lower(),
#            '+1:word.istitle()': word1.istitle(),
#            '+1:word.isupper()': word1.isupper(),
#            '+1:postag': postag1,
#            '+1:postag[:2]': postag1[:2],
#        })
#    else:
#        features['EOS'] = True
#
#    return features
#
#
#def sent2features(sent):
#    return [word2features(sent, i) for i in range(len(sent))]
    
df = load_data(max_=10, load_entities=False)
print(df)