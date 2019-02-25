#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:01:05 2019

@author: Daniel, Count Count
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import nltk
import sklearn
import numpy as np

from utils import flatten, plot_confusion_matrix, separate_labels_from_features

# -----------------------------------------------------------------------------
# Naive Bayes class
# -----------------------------------------------------------------------------

class Naive_Bayes(object):
    """
    Naive Bayes class
    """

    def __init__(self):
        """
        Constructor.
        """
        pass


    def fit_nltk(self, X):
        """
        Fit the model to the data using nltk.

        :param X:
        """
        self.clf_nltk = nltk.NaiveBayesClassifier.train(X)


    def predict_nltk(self, X):
        """
        Predict pos/ner tags.

        :param X:           data to predict labels for
        :return:            labels for the data
        """
        X_, y_ = separate_labels_from_features(X)
        
        y = []
        n_sent_correct = 0
        num_sent = len(y_)
        
        for i in range(num_sent):
            sentence_correct = True
            for j in range(len(y_[i])):
                y.append(self.clf_nltk.classify(X_[i][j]))
                if y_[i][j] != y[-1]:
                    sentence_correct = False
                    
            if sentence_correct == True:
                n_sent_correct += 1
                
        return flatten(y_)


    def evaluate_nltk(self, X):
        """
        Evaluates the naive bayes classifier
        """
        X_, y_ = separate_labels_from_features(X)
        
        y = []
        n_sent_correct = 0
        num_sent = len(y_)
        
        for i in range(num_sent):
            sentence_correct = True
            for j in range(len(y_[i])):
                y.append(self.clf_nltk.classify(X_[i][j]))
                if y_[i][j] != y[-1]:
                    sentence_correct = False
                    
            if sentence_correct == True:
                n_sent_correct += 1
                
        y_ = flatten(y_)
        print("F1 score:")
        print(sklearn.metrics.precision_recall_fscore_support(y_, y, average='micro'))
        print()
        print("Accuracy:")
        print(sklearn.metrics.accuracy_score(y_, y))
        print()
        print("Sentence level accuracy:")
        print(n_sent_correct / num_sent)
        print()
        print("F1 score per class:")
        print(sklearn.metrics.precision_recall_fscore_support(y_, y))
        print()
        print("Confusion matrix:")
        cfm = sklearn.metrics.confusion_matrix(y_, y)

        plot_confusion_matrix(cfm, np.unique(y_))
        
        print(np.unique(y_))
        print()
        print(print(np.unique(y)))
        
