#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:58:52 2017

@author: ray
"""

def tfidf(articles):
    
    length = len(articles)
    
    Word = {}
    bigdata = []
    for x in articles:
        for y in set(x):
            if y in Word:
                Word[y] +=1
            else:
                Word[y] = 1
                
            
    for x in articles: 
        dic = {}
        for word in set(x):
            
            count =  Counter(x)
            
            max_count = max(count.values())
            
            num = Word[word]
            
            idf = math.log(length/(1+num))
            
            tf = (0.5+(0.5*(count[word]/len(x))/max_count))
            
            dic[word] = idf*tf
            
        bigdata.append(dic)
        
    Data = pd.DataFrame(data=bigdata)
    Data = pd.DataFrame.as_matrix(Data.fillna(0))
    Data = np.matrix(Data)
    return Data