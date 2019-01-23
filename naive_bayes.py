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

# -----------------------------------------------------------------------------
# Naive Bayes class
# -----------------------------------------------------------------------------

class Naive_Bayes():
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
        return self.clf_nltk.predict(X)
    
    
    def evaluate_nltk(self, X):
        """
        Evaluates the naive bayes classifier
        """
        return nltk.classify.accuracy(self.clf_nltk, X)
