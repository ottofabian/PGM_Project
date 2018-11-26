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


df = load_data(max_=10, load_entities=False)
print(df)