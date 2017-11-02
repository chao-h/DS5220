import json
from nltk.stem.porter import *
#from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

stemmer = PorterStemmer()

m = RegexpTokenizer(r'\w+')

finFile = '../datas/bbc_stemmed.json'
foutFile = '../datas/bbc_removeStopwords.json'
stopwordFile = "../datas/stopwords.txt"

with open(finFile) as f:
    datas=json.load(f)

with open(stopwordFile) as f:
    stopwords = f.read().split(',')
    stopwords = [w[1:-1] for w in stopwords]
    stopwords = [stemmer.stem(w) for w in stopwords]
    stopwords = set(stopwords)

def removeStopwords(l):
    newl = []
    for w in l:
        if w not in stopwords and not w.isdigit():
            newl.append(w)
    return newl

for i in range(len(datas)):
    if(i%1000) == 0:
        print("Stopwords removed", i, "documents")
    data = datas[i]
    data['title'] = removeStopwords(data['title'])
    data['content'] = removeStopwords(data['content'])

with open(foutFile, 'w') as fout:
    json.dump(datas, fout, separators = (', \n', ': '))
