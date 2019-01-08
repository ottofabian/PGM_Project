#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:47:08 2019

@author: Clemens Biehl, Fabian Otto, Daniel Wehner
"""

import time
import pandas as pd

from hmm import HMM
from utils import preprocess_raw_data, load_data_list, train_test_split, show_misclassifications, split

preprocessing = False
load_entities = False


def main():
    """
    Main function.
    """
    # load data and preprocessing
    if preprocessing:
        # usually not needed since txt file was created
        data = preprocess_raw_data(max_=None, load_entities=load_entities)
    else:
        path = "data_POS.txt" if not load_entities else "data_NER.txt"
        data = load_data_list(path)

    # split data into training and test set    
    data_train, data_test = train_test_split(data, train_ratio=0.80)

    # fit model
    hmm = HMM()
    start_time = time.time()
    hmm.fit(data_train)
    print(f"Duration of training: {time.time() - start_time}")
    # evaluation
    # print(hmm.predict(
    #     ["This is a house .".split(),
    #      "This is Peter Parker .".split()]
    # ))

    hmm.evaluate(data_test[:100])
#
#    show_misclassifications(data_test, prediction)


# execute main program
if __name__ == "__main__":
   # main()
    
    df = pd.read_csv(
        "/Users/Daniel/Documents/PGM_Project/gmb-2.2.0/data/p51/d0431/en.tags",
        index_col=None,
        usecols=[0, 1 if not load_entities else 3],
        header=None,
        sep="\t",
        skip_blank_lines=False,
        dtype={
            0: str,
            1: str
        })
    
    mask = pd.isna(df[0])
    data = split(df, mask)
    print(data)