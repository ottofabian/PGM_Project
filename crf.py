#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:40:42 2019

@author: Clemens, Fabian Otto, Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sklearn
import scipy

import numpy as np
import sklearn_crfsuite
import sklearn_crfsuite.metrics as metrics

import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

from collections import Counter

from utils import flatten

# -----------------------------------------------------------------------------
# Conditional Random Field
# -----------------------------------------------------------------------------
from utils import plot_confusion_matrix


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

    def evaluate(self, X, y_):
        """
        Evaluates the trained crf model.
        
        :param X:   test data: [[(word, pos), (word, pos)], [...]]
        :param y:   test labels
        :return:    evaluation result
        """
        y = self.crf.predict(X)
        
        n_sent_correct = 0
        num_sent = len(X)

        for i in range(num_sent):
            sentence_correct = True
            for j in range(len(y_[i])):
                if y_[i][j] != y[i][j]:
                    sentence_correct = False
                    
            if sentence_correct == True:
                n_sent_correct += 1 

        labels = flatten(y_)
        y = flatten(y)
        
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

    def evaluate_sentence(self, X, y):
        """
        Evaluates the trained crf model on sentence level.
        
        :param X:   test data
        :param y:   test labels
        :return:    evaluation result / accuracy
        """
        y_pred = self.crf.predict(X)

        correct_count = 0
        num_sent = len(y)

        for i in range(num_sent):
            if y[i] == y_pred[i]:
                correct_count = correct_count + 1

        return correct_count / num_sent

    def optimize_hyperparameters(self, X, y, plot=True):
        """
        Optimizes CRF hyperparameters.
        
        :param X:       training data
        :param y:       training labels
        :param plot:    flag indicating if result should be plotted 
        """
        # define fixed parameters and parameters to search
        params_space = {
            "c1": scipy.stats.expon(scale=0.5),
            "c2": scipy.stats.expon(scale=0.05),
        }

        # use the same metric for evaluation
        f1_scorer = make_scorer(
            metrics.flat_f1_score,
            average="weighted", labels=self.crf.classes_
        )

        # search
        rs = RandomizedSearchCV(
            self.crf, params_space,
            cv=3,
            verbose=1,
            n_jobs=-1,
            n_iter=20,
            scoring=f1_scorer
        )

        rs.fit(X, y)
        print("best params:", rs.best_params_)
        print("best CV score:", rs.best_score_)
        print("model size: {:0.2f}M".format(rs.best_estimator_.size_ / 1000000))

        if plot:
            _x = [s["c1"] for s in rs.cv_results_["params"]]
            _y = [s["c2"] for s in rs.cv_results_["params"]]
            _c = rs.cv_results_["mean_test_score"]

            fig = plt.figure()
            fig.set_size_inches(12, 12)
            ax = plt.gca()
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel("C1")
            ax.set_ylabel("C2")
            ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
                min(_c), max(_c)
            ))

            ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0, 0, 0])

            print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

    def __print_transitions(self, trans_features):
        """
        Prints transitions.
        
        :param trans_features:
        """
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    def likely_transitions(self, n=20):
        """
        Prints likely transitions.
        
        :param n:   number of transitions
        """
        print("Top likely transitions:")
        self.__print_transitions(Counter(self.crf.transition_features_).most_common(n))

    def unlikely_transitions(self, n=20):
        """
        Prints unlikely transitions.
        
        :param n:   number of transitions
        """
        print("\nTop unlikely transitions:")
        self.__print_transitions(Counter(self.crf.transition_features_).most_common()[-n:])

    def __feature_importance(self, state_features):
        """
        Prints the importance of features.
        
        :param state_features:
        """
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def most_informative_features(self, n=30):
        """
        Prints the most informative features.
        
        :param n:   number of features
        """
        print("Top positive:")
        self.__feature_importance(Counter(self.crf.state_features_).most_common(n))

    def least_informative_features(self, n=30):
        """
        Prints the least informative features.
        
        :param n:   number of features
        """
        print("\nTop negative:")
        self.__feature_importance(Counter(self.crf.state_features_).most_common()[-n:])

    def classification_report(self, X, y):
        """
        Evaluates the trained crf model.
        
        :param X:   test data
        :param y:   test labels
        """
        y_pred = self.crf.predict(X)

        print(metrics.flat_classification_report(y, y_pred, digits=3))
