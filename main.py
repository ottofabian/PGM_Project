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
import pickle

from hmm import HMM
from crf import CRF
from naive_bayes import Naive_Bayes
from feature_maker import Feature_Maker
from utils import preprocess_raw_data, load_data_list, \
    train_test_split, show_misclassifications, separate_labels_from_features

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# global variables
preprocessing = False  # true: create txt file from data, false: load existing txt file with preprocessed data
load_entities = False  # true: ner, false: pos-tagging

model_type = "NB"


def main():
    """
    Main function.
    """
    # load data and preprocessing
    if preprocessing:
        # usually not needed since txt file was created
        data = preprocess_raw_data(max_=None, load_entities=load_entities, also_load_pos=True)
    else:
        path = "data_POS.txt" if not load_entities else "data_NER.txt"
        data = load_data_list(path)

    # split data into training and test set
    data_train, data_test = train_test_split(data, train_ratio=0.80)

    data_train = data_train
    data_test = data_test

    # print(data_train)

    feature_maker = Feature_Maker()

    if model_type == "HMM":
        # fit hidden markov model model
        # -------------------------------------------------------------------------
        hmm = HMM()
        start_time = time.time()
        hmm.fit(data_train)
        print(f"Duration of training: {time.time() - start_time}")

        # evaluation hmm
        # -------------------------------------------------------------------------
        # plot confusion matrix, calculate precision, recall, f1-score
        hmm.evaluate(data_test[:100])
        print("Sent Acc", hmm.evaluate_sentence(data_test[:100]))
        # show misclassifications

        features_test, labels_test = separate_labels_from_features(data_test)
        predictions = hmm.predict(features_test)
        show_misclassifications(data_test, predictions)

    elif model_type == "NB":
        # fit naive bayes model
        # -------------------------------------------------------------------------
        nb = Naive_Bayes()
        #    data_train_featurized = feature_maker.get_pos_features_nltk(data_train)
        data_train_featurized = feature_maker.get_ner_features_nltk(data_train)
        start_time = time.time()
        nb.fit_nltk(data_train_featurized)
        print(f"Duration of training: {time.time() - start_time}")

        # evaluation naive bayes
        # -------------------------------------------------------------------------
        data_test_featurized = feature_maker.get_pos_features_nltk(data_test)
        data_test_featurized = feature_maker.get_ner_features_nltk(data_test)
        print("Accuracy: ", nb.evaluate_nltk(data_test_featurized))
        # most informative features
        nb.clf_nltk.show_most_informative_features(50)

    elif model_type == "CRF":
        # fit crf model
        # -------------------------------------------------------------------------
        features_train = feature_maker.get_ner_features_crf(data_train)
        features_test = feature_maker.get_ner_features_crf(data_test)
        X, y = separate_labels_from_features(features_train)
        X_test, y_test = separate_labels_from_features(features_test)

        crf = CRF()
        crf.fit(X, y)

        print("Done with CRF learning")
        with open("crf_ner", "wb") as f:
            pickle.dump(crf, f)

        print(crf.evaluate(X_test, y_test))

        crf.optimize_hyperparameters(X, y, plot=True)
        crf.most_informative_features(30)
        crf.least_informative_features(30)
        crf.likely_transitions()
        crf.unlikely_transitions()
        print("Sent Acc:", crf.evaluate_sentence(X, y))
        crf.classification_report(X, y)


# execute main program
if __name__ == "__main__":
    main()
