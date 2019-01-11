#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:45:57 2019

@author: Clemens, Daniel
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import re
import numpy as np

from utils import flatten


# -----------------------------------------------------------------------------
# Feature_Maker
# -----------------------------------------------------------------------------

class Feature_Maker():
    """
    Feature Maker class.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    

    def _wordshape(self, t):
        """
        Convert token to shape string
        E.g. 35-Year -> dd-Xxxx
        
        :param t:       input token
        :return:        shape string
        """
        t = re.sub("[A-Z]", "X", t)
        t = re.sub("[a-z]", "x", t)
        
        return re.sub("[0-9]", "d", t)
    
    
    def get_pos_features_nltk(self, X):
        """
        Generate feature set for POS tagging
        
        :param X:       list of tuples [(w1, t1), (w2, t2), ...]
        :return:        dict of features
        """
        X_ = []
        X = flatten(X)
        
        for i, x in enumerate(X):
            word = x[0]
            postag = x[1]
            
            instance = ({
                "word": word,
                "lowercasedword": word.lower(),
                "prefix1": word[0],
                "prefix2": word[:2],
                "prefix3": word[:3],
                "suffix1": word[-1],
                "suffix2": word[-2:],
                "suffix3": word[-3:],
                "capitalization": word[0].isupper(),
                "shape": self._wordshape(word),
                "previousword": X[i-1][0] if i > 1 else "<BEGIN>",
                "nextword": X[i+1][0] if i < len(X)-1 else "<END>"
            }, postag)
            X_.append(instance)
            
        return X_
    
    
    def get_pos_features_sklearn(self, X):
        """
        Generate feature set for POS tagging
        
        :param X:       list of tuples [(w1, t1), (w2, t2), ...]
        :return:        dict of features
        """
        X_ = []
        y_ = []
        X = flatten(X)
        
        for i, x in enumerate(X):
            word = x[0]
            postag = x[1]
            
            features = [
                word,                                   # word
                word.lower(),                           # lowercaseword
                word[0],                                # prefix1
                word[:2],                               # prefix2
                word[:3],                               # prefix3
                word[-1],                               # suffix1
                word[-2:],                              # suffix2
                word[-3:],                              # suffix3
                word[0].isupper(),                      # capitalization
                self._wordshape(word),                  # shape
                X[i-1][0] if i > 1 else "<BEGIN>",      # previousword
                X[i+1][0] if i < len(X)-1 else "<END>"  # nextword
            ]
            X_.append(features)
            y_.append(postag)
            
        return np.asarray(X_), np.asarray(y_)
    
    
    def get_ner_features_nltk(self, X):
        """
        Generate feature set for NER tagging
        
        @param X: list of tuples [(word1, postag1), (word2, postag2), ...]
        @returns: dict of features
        """
        X_ = []
        
        for i, x in enumerate(X):
            word = x[0]
            postag = x[1]
        
            features = {
                "bias": 1.0,
                "lowercasedword": word.lower(),
                "prefix1": word[0],
                "prefix2": word[:2],
                "prefix3": word[:3],
                "suffix1": word[-1],
                "suffix2": word[-2:],
                "suffix3": word[-3:],
                "isuppercase": word.isupper(),
                "istitle": word.istitle(),
                "isdigit": word.isdigit(),
                "postag": postag,
                "basepos": postag[:2],
                "shape": self._wordshape(word)
            }
            
            if i > 0:
                word1 = X[i-1][0]
                postag1 = X[i-1][1]
                features.update({
                    "-1:lowercasedword": word1.lower(),
                    "-1:istitle": word1.istitle(),
                    "-1:isuppercase": word1.isupper(),
                    "-1:postag": postag1,
                    "-1:basepos": postag1[:2],
                    "-1:shape": self._wordshape(word1)
                })
            else:
                features['BOS'] = True
        
            if i < len(X) - 1:
                word1 = X[i+1][0]
                postag1 = X[i+1][1]
                features.update({
                    "+1:lowercasedword": word1.lower(),
                    "+1:istitle": word1.istitle(),
                    "+1:isuppercase": word1.isupper(),
                    "+1:postag": postag1,
                    "+1:basepos": postag1[:2],
                    "+1:shape": self._wordshape(word1)
                })
            else:
                features["EOS"] = True
                
            X_.append(features)
    
        return X_

    