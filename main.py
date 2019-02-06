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
import math
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

from hmm import HMM
from crf import CRF
from naive_bayes import Naive_Bayes
from feature_maker import Feature_Maker
from utils import preprocess_raw_data, load_data_list, flatten, \
    train_test_split, show_misclassifications, separate_labels_from_features

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# global variables
preprocessing = False  # true: create txt file from data, false: load existing txt file with preprocessed data
load_entities = True   # true: ner, false: pos-tagging

model_type = "CRF"
most_informative_features = 50


#def main():
#    """
#    Main function.
#    """

#global preprocessing
#global load_entities
#global model_type
#global most_informative_features

# load data and preprocessing
if preprocessing:
    # usually not needed since txt file was created
    data = preprocess_raw_data(max_=None, load_entities=load_entities, also_load_pos=True)
else:
    path = "data_POS.txt" if not load_entities else "data_NER.txt"
    data = load_data_list(path)

#data = data[:math.ceil(len(data)/4)]
    
#fig = plt.figure()
#fig.set_size_inches(12, 12)
#fd = FreqDist([t[1] for sent in data for t in sent])
#total = fd.N()
#for word in fd:
#    fd[word] /= float(total)
#
#fd.plot()

# split data into training and test set
data_train, data_test = train_test_split(data, train_ratio=0.01)

del data

# print(data_train)

feature_maker = Feature_Maker()

if model_type == "HMM":
    # fit hidden markov model model
    # -------------------------------------------------------------------------
    if load_entities is True:
        data_train = [[(t[0], t[2]) for t in sent] for sent in data_train]
        data_test = [[(t[0], t[2]) for t in sent] for sent in data_test]
    else:
        data_train = [[(t[0], t[1]) for t in sent] for sent in data_train]
        data_test = [[(t[0], t[1]) for t in sent] for sent in data_test]
    
    hmm = HMM()
    start_time = time.time()
    hmm.fit(data_train)
    print(f"Duration of training: {time.time() - start_time}")

    # evaluation hmm
    # -------------------------------------------------------------------------
    # plot confusion matrix, calculate precision, recall, f1-score
    hmm.evaluate(data_test)
    # show misclassifications
    #features_test, labels_test = separate_labels_from_features(data_test)
    #predictions = hmm.predict(features_test)
    #print("GET READY FOR SPAM!!!")
    #show_misclassifications(data_test, predictions)

elif model_type == "NB":
    # fit naive bayes model
    # -------------------------------------------------------------------------
    nb = Naive_Bayes()
    data_train_featurized = feature_maker.get_pos_features_nltk(
        data_train) if not load_entities else feature_maker.get_ner_features_nltk(data_train)
    
    data_train_featurized = flatten(data_train_featurized)
    start_time = time.time()
    nb.fit_nltk(data_train_featurized)
    print(f"Duration of training: {time.time() - start_time}")

    # evaluation naive bayes
    # -------------------------------------------------------------------------
    data_test_featurized = feature_maker.get_pos_features_nltk(
        data_test) if not load_entities else feature_maker.get_ner_features_nltk(data_test)
    
    nb.evaluate_nltk(data_test_featurized)
    print()
    # most informative features
    nb.clf_nltk.show_most_informative_features(most_informative_features)

elif model_type == "CRF":
    # fit crf model
    # -------------------------------------------------------------------------
    features_train = feature_maker.get_pos_features_crf(
        data_train) if not load_entities else feature_maker.get_ner_features_crf(data_train)
    features_test = feature_maker.get_pos_features_crf(
        data_test) if not load_entities else feature_maker.get_ner_features_crf(data_test)
    
    del data_train
    del data_test
    
    X, y = separate_labels_from_features(features_train)
    X_test, y_test = separate_labels_from_features(features_test)
    
    del features_train
    del features_test

#    crf = CRF(c1=0.3684, c2=0.0125) # embeddings
    crf = CRF(c1=0.4043, c2=0.1653) # hand-crafted features
    crf.fit(X, y)

    print("Done with CRF learning")
    with open("crf_ner", "wb") as f:
        pickle.dump(crf, f)

    print(crf.evaluate(X_test, y_test))

#    crf.optimize_hyperparameters(X, y, plot=True)
    crf.most_informative_features(most_informative_features)
    crf.least_informative_features(most_informative_features)
    crf.likely_transitions()
    crf.unlikely_transitions()
    crf.classification_report(X, y)


## execute main program
#if __name__ == "__main__":
#    main()
