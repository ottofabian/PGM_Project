#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:47:08 2019

@author: Clemens Biehl, Fabian Otto, Daniel Wehner
"""

import time

from hmm import HMM
from utils import preprocess_raw_data, load_data_list, train_test_split, show_misclassifications


def main():
    """
    Main function.
    """
    preprocessing = False
    # load data and preprocessing
    if preprocessing:
        # usually not needed since txt file was created
        data = preprocess_raw_data(max_=None, load_entities=False)
    else:
        data = load_data_list("data.txt")
    
    # split data into training and test set    
    data_train, data_test = train_test_split(data, train_ratio=0.80)
    
    # fit model
    hmm = HMM()
    start_time = time.time()
    hmm.fit(data_train)
    print(f"Duration of training: {time.time() - start_time}")
    # evaluation
    print(hmm.predict(
        ["This is a house .".split(),
         "This is Peter Parker .".split()]
    ))
#    hmm.evaluate(data_test)
#    
#    show_misclassifications(data_test, prediction)
    

# execute main program
if __name__ == "__main__":
    main()
