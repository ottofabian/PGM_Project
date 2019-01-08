#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:47:08 2019

@author: Clemens, Fabian Otto, Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import time

from hmm import HMM
from utils import preprocess_raw_data, load_data_list, \
    train_test_split, show_misclassifications, separate_labels_from_features

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# global variables
preprocessing = True
load_entities = True


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

    # fit hidden markov model model
    # -------------------------------------------------------------------------
    hmm = HMM()
    start_time = time.time()
    hmm.fit(data_train)
    print(f"Duration of training: {time.time() - start_time}")

    # evaluation
    # -------------------------------------------------------------------------
    # plot confusion matrix, calculate precision, recall, f1-score
    hmm.evaluate(data_test[:100])
    # show misclassifications


#    features_test, labels_test = separate_labels_from_features(data_test)
#    predictions = hmm.predict(features_test)
#    show_misclassifications(data_test, predictions)


# execute main program
if __name__ == "__main__":
    main()

    # df = pd.read_csv(
    #     "./gmb-2.2.0/data/p51/d0431/en.tags",
    #     index_col=None,
    #     usecols=[0, 1 if not load_entities else 3],
    #     header=None,
    #     sep="\t",
    #     skip_blank_lines=False,
    #     quotechar="\"",
    #     engine='python',
    #     doublequote=False,
    #     dtype={
    #         0: str,
    #         1: str,
    #         # 3: str
    #     })
    # df.replace("\tLQU\t", '"', inplace=True)
    # df.replace("\tRQU\t", '"', inplace=True)
    # df.replace("[]", "QU", inplace=True)
    # df.replace('None', np.nan, inplace=True)
    #
    # # print(df.iloc[19])
    # # df.fillna(value=pd.np.nan, inplace=True)
    # # print(type(df.iloc[19, 0]))
    # mask = pd.isna(df[0])
    # print(mask.iloc[19])
    # data = split(df, mask)
    # print(df.iloc[94].values)
    # print(df.iloc[1].values)
    # print(data)
