#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:39:19 2019

@author: Clemens, Fabian Otto, Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import nltk
import sklearn
import numpy as np

from nltk.tag import hmm
from utils import plot_confusion_matrix, separate_labels_from_features


# -----------------------------------------------------------------------------
# CustomTagger class
# -----------------------------------------------------------------------------

class CustomTagger(nltk.tag.HiddenMarkovModelTagger):
    """
    Custom Tagger class.
    """

    def __init__(self, symbols, states, transitions, outputs, priors):
        """
        Constructor.
        
        :param symbols:
        :param states:
        :param transitions:
        :param outputs:
        :param priors:
        """
        # call super constructor
        super().__init__(symbols, states, transitions, outputs, priors)

    def _tag(self, unlabeled_sequence):
        """
        Tags the data.
        
        :param unlabeled_sequence:
        :return:
        """
        path = self._best_path(unlabeled_sequence)
        return unlabeled_sequence, path


# -----------------------------------------------------------------------------
# Hidden Markov Model class
# -----------------------------------------------------------------------------

class HMM(nltk.tag.HiddenMarkovModelTrainer):
    """
    Hidden Markov Model class.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
        self.trainer = hmm.HiddenMarkovModelTrainer()
        self.tagger = None

    def fit(self, X):
        """
        Fit model using data.
        
        :param X:           training data in the form: [[(w1, t1), (w2, t2), ...], [...], ...]
        """
        self.tagger = self.train_supervised(X)

    def predict(self, X):
        """
        Predict pos/ner tags.
        
        :param X:           data to predict labels for
        :return:            words with labels in the form: [[(w1, t1), (w2, t2), ...], [...], ...]
        """
        prediction = []

        for sent in X:
            _, y = self.tagger.tag(sent)
            prediction.append(y)

        return prediction

    def evaluate(self, X):
        """
        Evaluate the classifier.
        
        :param X:           data to test on
        :return:            evaluation results
        """
        features, labels = separate_labels_from_features(X)

        # get predictions for data
        y = self.predict(features)
        
        n_sent_correct = 0
        num_sent = len(y)

        for i in range(len(labels)):
            if labels[i] == y[i]:
                n_sent_correct += 1

        labels = nltk.flatten(labels)
        y = nltk.flatten(y)
        
        print("F1 score:")
        print(sklearn.metrics.precision_recall_fscore_support(labels, y, average='micro'))
        print()
        print("Accuracy:")
        print(sklearn.metrics.accuracy_score(labels, y))
        print()
        print("Sentence level accuracy:")
        print(n_sent_correct / num_sent)
        print()
        print("F1 score per class:")
        print(sklearn.metrics.precision_recall_fscore_support(labels, y))
        print()
        print("Confusion matrix:")
        cfm = sklearn.metrics.confusion_matrix(labels, y)

        plot_confusion_matrix(cfm, np.unique(labels))


    def train_supervised(self, labelled_sequences, estimator=None):
        """
        Trains model in supervised fashion.
        
        :param labelled_sequences:
        :param estimator:
        :return:
        """
        tagger = super().train_supervised(labelled_sequences, estimator)
        return CustomTagger(self._symbols, self._states, tagger._transitions, tagger._outputs, tagger._priors)

    def train_unsupervised(self, unlabeled_sequences, update_outputs=True, **kwargs):
        """
        Trains model in unsupervised fashion.
        
        :param unlabeled_sequences:
        :param update_outputs:
        """
        raise NotImplementedError()
