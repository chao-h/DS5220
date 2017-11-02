import numpy as np
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import *

category=[]
content=[]

with open('../datas/bbc.json') as f:
    datas=json.load(f)
    for data in datas:
        category.append(data['category'])
        content.append(data['content'])

vect=TfidfVectorizer()

x=vect.fit_transform(content)
y=category

mnb=MultinomialNB()
mnb.fit(x,y)

predicts = [mnb.predict(vect.transform([data['content']])) for data in datas]
right = 0
for i in range(len(predicts)):
    if predicts[i][0] == datas[i]['category']:
        right += 1

right

while True:
    n=raw_input('Please enter topic: ')
    if n == '':
        break

    print mnb.predict(vect.transform([n]))[0]