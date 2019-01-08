#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:39:19 2019

@author: Clemens Biehl, Fabian Otto, Daniel Wehner
"""

from nltk.tag import hmm


class HMM():
    """
    Hidden Markov Model class.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        self.trainer = hmm.HiddenMarkovModelTrainer()
    
    
    def fit(self, X_train):
        """
        Fit model using data.
        
        :param X_train:     training data in the form: [[(w1, t1), (w2, t2), ...], [...], ...]
        """
        self.tagger = self.trainer.train_supervised(X_train)
    
    
    def predict(self, X):
        """
        Predict pos/ner tags.
        
        :param X:           data to predict labels for
        :return:            words with labels in the form: [[(w1, t1), (w2, t2), ...], [...], ...]
        """
        return self.tagger.tag(X)
    
    
    def evaluate(self, X):
        """
        Evaluate the classifier.
        
        :param X:           data to test on
        :return:            evaluation results
        """
        self.tagger.test(X)
    