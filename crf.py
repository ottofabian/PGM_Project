#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:40:42 2019

@author: Clemens, Fabian Otto, Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import sklearn_crfsuite
import sklearn_crfsuite.metrics as metrics

# -----------------------------------------------------------------------------
# Conditional Random Field
# -----------------------------------------------------------------------------

class CRF():
    """
    Conditional Random Field class.
    """

    def __init__(self, c1=0.1, c2=0.1, max_iter=100, all_possible_transitions=True, algorithm="lbfgs"):
        """
        Constructor.
        """
        self.crf = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iter,
            all_possible_transitions=all_possible_transitions
        )
        

    def fit(self, X, y):
        """
        Fits the model.

        :param X:   training data
        :param y:   training labels
        """
        self.crf.fit(X, y)
        

    def predict(self, X):
        """
        Predicts the labels.

        :param X:   data to predict labels for
        :return:    labels for the data
        """
        return self.crf.predict(X)
    
    
    def evaluate(self, X, y):
        """
        Evaluates the trained crf model.
        
        :param X:
        """
        y_pred = self.crf.predict(X)
        
        return metrics.flat_f1_score(y, y_pred, average='weighted')
    
    
    def classification_report(self, X, y):
        """
        Evaluates the trained crf model.
        
        :param X:
        """
        y_pred = self.crf.predict(X)
        
        print(metrics.flat_classification_report(y, y_pred, digits=3))
        