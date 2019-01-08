#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:01:05 2019

@author: Daniel
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from sklearn.naive_bayes import MultinomialNB


# -----------------------------------------------------------------------------
# Naive Bayes class
# -----------------------------------------------------------------------------

class Naive_Bayes():
    """
    Naive Bayes class
    """
    
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        """
        Constructor.
        """
        
        self.clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
    
    
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        :param X: training data
        :param y: training labels
        """
        
        self.clf.fit(X, y)
    
    
    def predict(self, X):
        """
        Predict pos/ner tags.
        
        :param X:           data to predict labels for
        :return:            labels for the data
        """
        
        return self.clf.predict(X)
