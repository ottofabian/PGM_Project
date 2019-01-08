import re

def shape(t):
    t = re.sub('[A-Z]', 'X', t)
    t = re.sub('[a-z]', 'x', t)
    
    return re.sub('[0-9]', 'd', t)


def get_pos_features(X):
    """
    Generate feature set for POS tagging
    
    @param X: list of tuples [(word1, postag1), (word2, postag2), ...]
    @returns: dict of features
    """
    
    X_ = []
    
    for i, x in enumerate(X):
        features = {"word": x,
                    "lowercasedword": x.lower(),
                    "prefix1": x[0],
                    "prefix2": x[:2],
                    "prefix3": x[:3],
                    "suffix1": x[-1],
                    "suffix2": x[-2:],
                    "suffix3": x[-3:],
                    "capitalization": x[0].isupper(),
                    "shape": shape(x),
                    "previousword": X[i-1] if i > 1 else "<BEGIN>",
                    "nextword": X[i+1] if i < len(X)-1 else "<END>"}
        
        X_.append(features)
        
    return X_


def get_ner_features(X):
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
            "shape": shape(word)
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
            })
        else:
            features['EOS'] = True
            
        X_.append(features)

    return X_
