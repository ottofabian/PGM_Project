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

from nltk.stem.snowball import SnowballStemmer

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
        self.stemmer = SnowballStemmer("english")

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


    def _wordshape_short(self, t):
        """
        Convert token to short shape string
        E.g. 35-Year -> d-Xx

        :param t:       input token
        :return:        shape string
        """
        t = re.sub("[A-Z]+", "X", t)
        t = re.sub("[a-z]+", "x", t)

        return re.sub("[0-9]+", "d", t)


    def get_pos_features_nltk(self, X):
        """
        Generate feature set for POS tagging

        :param X:       list of tuples [(w1, t1), (w2, t2), ...]
        :return:        dict of features
        """
        X_ = []

        for sent in X:
            sent_instances = []
            for i, x in enumerate(sent):
                word = x[0]
                postag = x[1]

                instance = ({
                                "word": word,
                                "lowercasedword": word.lower(),
                                "stem": self.stemmer.stem(word),
#                                 "prefix1": word[0],
#                                 "prefix2": word[:2],
#                                 "prefix3": word[:3],
#                                 "suffix1": word[-1],
#                                 "suffix2": word[-2:],
#                                 "suffix3": word[-3:],
#                                 "capitalization": word[0].isupper(),
#                                 "shape": self._wordshape(word)
                                 "previousword": sent[i - 1][0] if i > 1 else "<BEGIN>",
                                 "nextword": sent[i + 1][0] if i < len(sent) - 1 else "<END>"
                            }, postag)

                sent_instances.append(instance)

            X_.append(sent_instances)

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
                word,  # word
                word.lower(),  # lowercaseword
                self.stemmer.stem(word),  # stem
                word[0],  # prefix1
                word[:2],  # prefix2
                word[:3],  # prefix3
                word[-1],  # suffix1
                word[-2:],  # suffix2
                word[-3:],  # suffix3
                word[0].isupper(),  # capitalization
                self._wordshape(word),  # shape
                X[i - 1][0] if i > 1 else "<BEGIN>",  # previousword
                X[i + 1][0] if i < len(X) - 1 else "<END>"  # nextword
            ]
            X_.append(features)
            y_.append(postag)

        return np.asarray(X_), np.asarray(y_)


    def get_pos_features_crf(self, X):
        """
        Generate feature set for POS tagging

        :param X:       list of tuples [(w1, t1), (w2, t2), ...]
        :return:        dict of features
        """
        X_ = []

        for sent in X:
            sent_instances = []
            for i, x in enumerate(sent):
                word = x[0]
                postag = x[1]

                instance = ({
                    "word": word,
                    "lowercasedword": word.lower(),
                    "stem": self.stemmer.stem(word),
                    "prefix1": word[0],
                    "prefix2": word[:2],
                    "prefix3": word[:3],
                    "suffix1": word[-1],
                    "suffix2": word[-2:],
                    "suffix3": word[-3:],
                    "capitalization": word[0].isupper(),
                    "shape": self._wordshape(word),
                    "previousword": sent[i-1][0] if i > 0 else "<BEGIN>",
                    "nextword": sent[i+1][0] if i < len(sent)-1 else "<END>"
                }, postag)

                sent_instances.append(instance)

            X_.append(sent_instances)

        return X_


    def get_ner_features_nltk(self, X):
        """
        Generate feature set for NER tagging

        @param X: list of tuples [(w1, pos1, t1), (w2, pos2, t2), ...]
        @returns: dict of features
        """
        X_ = []

        if len(X[0][0]) < 3:
            raise ValueError("Expected list of tuples in form [(w1, pos1, t1), (w2, pos2, t2), ...].")

        for sent in X:
            sent_instance = []
            for i, x in enumerate(sent):
                word = x[0]
                postag = x[1]
    
                instance = ({
                                "bias": 1.0,
                                "word": word,
                                "lowercasedword": word.lower(),
                                "stem": self.stemmer.stem(word),
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
                            }, x[2])
    
                if i > 0:
                    word1 = sent[i - 1][0]
                    postag1 = sent[i - 1][1]
                    instance[0].update({
                        "-1:lowercasedword": word1.lower(),
                        "-1:istitle": word1.istitle(),
                        "-1:isuppercase": word1.isupper(),
                        "-1:postag": postag1,
                        "-1:basepos": postag1[:2],
                        "-1:shape": self._wordshape(word1)
                    })
                else:
                    instance[0]["BOS"] = True
    
                if i < len(sent) - 1:
                    word1 = sent[i + 1][0]
                    postag1 = sent[i + 1][1]
                    instance[0].update({
                        "+1:lowercasedword": word1.lower(),
                        "+1:istitle": word1.istitle(),
                        "+1:isuppercase": word1.isupper(),
                        "+1:postag": postag1,
                        "+1:basepos": postag1[:2],
                        "+1:shape": self._wordshape(word1)
                    })
                else:
                    instance[0]["EOS"] = True
                    
                sent_instance.append(instance)
    
            X_.append(sent_instance)

        return X_

    def get_ner_features_crf(self, X):
        """
        Generate feature set for NER tagging

        @param X: list of tuples [(w1, pos1, t1), (w2, pos2, t2), ...]
        @returns: dict of features
        """
        X_ = []

        if len(X[0][0]) < 3:
            raise ValueError("Expected list of tuples in form [(w1, pos1, t1), (w2, pos2, t2), ...].")

        for sent in X:
            sent_instances = []
            for i, x in enumerate(sent):
                word = x[0]
                postag = x[1]

                instance = ({
                                "bias": 1.0,
                                "word": word,
                                "lowercasedword": word.lower(),
                                "stem": self.stemmer.stem(word),
                                "prefix1": word[0],
                                "prefix2": word[:2],
                                "prefix3": word[:3],
                                "suffix1": word[-1],
                                "suffix2": word[-2:],
                                "suffix3": word[-3:],
                                "isFirst": i == 0,
                                "isLast": i == len(sent) - 1,
                                "hasHyphen": "-" in word,
                                "hasPeriod": "." in word,
                                "isuppercase": word.isupper(),
                                "istitle": word.istitle(),
                                "isdigit": word.isdigit(),
                                "postag": postag,
                                "basepos": postag[:2],
                                "shape": self._wordshape(word)
                            }, x[2])

                if i > 0:
                    word1 = sent[i - 1][0]
                    postag1 = sent[i - 1][1]
                    instance[0].update({
                        "-1:lowercasedword": word1.lower(),
                        "-1:istitle": word1.istitle(),
                        "-1:isuppercase": word1.isupper(),
                        "-1:postag": postag1,
                        "-1:basepos": postag1[:2],
                        "-1:shape": self._wordshape(word1)
                    })
                else:
                    instance[0]["BOS"] = True

                if i < len(sent) - 1:
                    word1 = sent[i + 1][0]
                    postag1 = sent[i + 1][1]
                    instance[0].update({
                        "+1:lowercasedword": word1.lower(),
                        "+1:istitle": word1.istitle(),
                        "+1:isuppercase": word1.isupper(),
                        "+1:postag": postag1,
                        "+1:basepos": postag1[:2],
                        "+1:shape": self._wordshape(word1)
                    })
                else:
                    instance[0]["EOS"] = True

                sent_instances.append(instance)

            X_.append(sent_instances)

        return X_
