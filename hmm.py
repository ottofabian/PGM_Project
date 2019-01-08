#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:39:19 2019

@author: Clemens Biehl, Fabian Otto, Daniel Wehner
"""

import nltk

from nltk.tag import hmm


class HMM(nltk.tag.HiddenMarkovModelTagger):
    """
    Hidden Markov Model class.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        super(self, HMM).__init__()
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
        prediction = []
        
        for sent in X:
            prediction.append(self.tagger.tag(sent))
            
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
        prediction = self.predict(unlabeled_data)
                
                
    