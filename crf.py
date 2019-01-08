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


# -----------------------------------------------------------------------------
# Conditional Random Field
# -----------------------------------------------------------------------------

class CRF(object):
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

        :param X: training data
        :param y: training labels
        """
        self.crf.fit(X, y)
        

    def predict(self, X):
        """
        Predicts the labels.

        :param X:
        """
        return self.crf.predict(X)
