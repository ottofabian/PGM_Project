#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:47:08 2019

@author: Clemens Biehl, Fabian Otto, Daniel Wehner
"""

from utils import load_data
from hmm import HMM


def main():
    data_train = load_data(max_=10, load_entities=False)
    hmm = HMM()
    hmm.fit(data_train)
    print(hmm.predict("This is a house.".split()))
    hmm.evaluate(data_train)
    

# execute main program
if __name__ == "__main__":
    main()
    
