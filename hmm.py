#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:39:19 2019

@author: Clemens Biehl, Fabian Otto, Daniel Wehner
"""
import numpy as np
import sklearn

import nltk

from nltk.tag import hmm

from utils import plot_confusion_matrix


class CustomTagger(nltk.tag.HiddenMarkovModelTagger):

    def __init__(self, symbols, states, transitions, outputs, priors):
        super().__init__(symbols, states, transitions, outputs, priors)

    def _tag(self, unlabeled_sequence):
        path = self._best_path(unlabeled_sequence)
        return unlabeled_sequence, path


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
        
        :param X:     training data in the form: [[(w1, t1), (w2, t2), ...], [...], ...]
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
        #        self.tagger.test(X)
        unlabeled_data = []
        labels = []

        # separate data from labels
        for sent in X:
            sub_unlabeled_data = []
            sub_labels = []
            for (w, t) in sent:
                sub_unlabeled_data.append(w)
                sub_labels.append(t)
            unlabeled_data.append(sub_unlabeled_data)
            labels.append(sub_labels)

        # get predictions for data
        y = self.predict(unlabeled_data)

        labels = nltk.flatten(labels)
        y = nltk.flatten(y)

        print(sklearn.metrics.precision_recall_fscore_support(labels, y))
        cfm = sklearn.metrics.confusion_matrix(labels, y)

        plot_confusion_matrix(cfm, np.unique(labels))

    def train_supervised(self, labelled_sequences, estimator=None):
        tagger = super().train_supervised(labelled_sequences, estimator)
        return CustomTagger(self._symbols, self._states, tagger._transitions, tagger._outputs, tagger._priors)

    def train_unsupervised(self, unlabeled_sequences, update_outputs=True, **kwargs):
        raise NotImplementedError()
