# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import nltk
from nltk.stem.porter import *
stemmer = PorterStemmer()
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
with open('bbc.json') as f:
    datas=json.load(f)
    
m = RegexpTokenizer(r'\w+')
for data in datas:
    l1 = m.tokenize(data["title"])
    l2 = m.tokenize(data["content"])
    for i in range(len(l1)):
        l1[i] = stemmer.stem(l1[i])
    for i in range(len(l2)):
        l2[i] = stemmer.stem(l2[i])
    data["title"] = " ".join(l1) 
    data["content"] = " ".join(l2) 
    

