import json
from nltk.stem.porter import *
#from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

finFile = 'bbc_removeNonPrintable.json'
foutFile = 'bbc_stemmed.json'

with open('bbc_removeNonPrintable.json') as f:
    datas=json.load(f)

stemmer = PorterStemmer()

m = RegexpTokenizer(r'\w+')

cnt=0
for data in datas:
    l1 = m.tokenize(data["title"])
    l2 = m.tokenize(data["content"])
    l1 = [stemmer.stem(l) for l in l1]
    l2 = [stemmer.stem(l) for l in l2]
    data["title"] = l1
    data["content"] = l2 
    #
    cnt += 1
    if((cnt%1000) == 0):
        print("Stemmed",cnt,"documents")

with open(foutFile, 'w') as fout:
    json.dump(datas, fout, separators = (', \n', ': '))

