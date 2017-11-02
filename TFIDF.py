#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:23:36 2017

@author: ray
"""
d0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
d1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."

import math
from textblob import TextBlob

class TFIDF:
    
    def __init__(self, word):
        self.word = word
        
    # term frequency
    def term_freq(self, word, article):
        article = TextBlob(article)
        return article.words.count(word)/len(article.split(" "))
    
    #Normalized term frequency
    def tf(self, article):
        max_count = max([self.term_freq(t, article) for t in article])
        return (0.5+ (0.5 * self.term_freq(self.word, article)/max_count))
                
    #number of articles that contain the word   
    #articles: list of articles(list of strings)
    def num_contain(self, articles):
        return sum(1 for article in articles if self.word in article)
    
    #inverse document frequency
    
    def idf(self, articles):
        return math.log(len(articles))/(1+self.num_contain(articles))
    
    def tfidf(self, article, articles):
        return self.tf(article) * self.idf(articles)

aaa= TFIDF("China")
aaa.term_freq("China",d0)
aaa.tf(d0)
aaa.num_contain([d0,d1])
aaa.idf([d0,d1])
aaa.tfidf(d0,[d0,d1])