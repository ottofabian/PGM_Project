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
            skip_blank_lines=False,
            dtype={
                0: str,
                1: str
            })
        list_.append(df)
        
    frame = pd.concat(list_, axis=0, ignore_index=True)
    mask = pd.isna(frame[0])
    data = split(frame, mask)
    
    print("Finished loading data.")
    return data


def split(df, mask):
    """
    Split data frame into sentences.
    
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