#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:45:57 2019

@author: Daniel
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# tbd


# -----------------------------------------------------------------------------
# Feature_Maker
# -----------------------------------------------------------------------------

class Feature_Maker():
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def extract_features(self, X):
        """
        Extracts the features.
        
        :param X:       raw data
        :return:        extracted features
        """
        pass


#def word2features(sent, i):
#    word = sent[i][0]
#    postag = sent[i][1]
#
#    features = {
#        'bias': 1.0,
#        'word.lower()': word.lower(),
#        'word[-3:]': word[-3:],
#        'word.isupper()': word.isupper(),
#        'word.istitle()': word.istitle(),
#        'word.isdigit()': word.isdigit(),
#        'postag': postag,
#        'postag[:2]': postag[:2],
#    }
#    if i > 0:
#        word1 = sent[i-1][0]
#        postag1 = sent[i-1][1]
#        features.update({
#            '-1:word.lower()': word1.lower(),
#            '-1:word.istitle()': word1.istitle(),
#            '-1:word.isupper()': word1.isupper(),
#            '-1:postag': postag1,
#            '-1:postag[:2]': postag1[:2],
#        })
#    else:
#        features['BOS'] = True
#
#    if i < len(sent)-1:
#        word1 = sent[i+1][0]
#        postag1 = sent[i+1][1]
#        features.update({
#            '+1:word.lower()': word1.lower(),
#            '+1:word.istitle()': word1.istitle(),
#            '+1:word.isupper()': word1.isupper(),
#            '+1:postag': postag1,
#            '+1:postag[:2]': postag1[:2],
#        })
#    else:
#        features['EOS'] = True
#
#    return features
#
#
#def sent2features(sent):
#    return [word2features(sent, i) for i in range(len(sent))]