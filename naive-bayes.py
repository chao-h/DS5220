'''
Naive Bayes with following algorithm:
p(c|d) = log(p(c)) + sum(log(p(w|c)))
where p(w|c) = (Wcw+1) / sum(Wcw+1))

c: category (class)
d: document
w: word
Wcw: number of word w in category c
sum(Wcw): total number of words in category c
'''

from collections import defaultdict
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from math import log

class NaiveBayes:
    def __init__(self):
        # total number of total documents
        self.document_count         = 0
        # all possible classes
        self.class_set              = set()
        # total number of classes
        self.class_count            = 0
        # number of documents in each classes
        self.class_document_count   = defaultdict(lambda: 0)
        # total number of words in each classes [category][word]
        self.class_total_word_count = defaultdict(lambda: 0) 
        # number of each words in each classes [word][category]
        self.word_class_count       = defaultdict(lambda: defaultdict(lambda: 0))
        # total number of each words
        self.word_count             = defaultdict(lambda: 0)
        # word score = P(t|C), set default value to eliminate zeros
        self.word_score             = defaultdict(lambda: defaultdict(lambda: 0))
        # set of individual words in each classes
        self.class_word_set         = defaultdict(lambda: set())
        #
        # Stemmer
        self.stemmer = PorterStemmer()
        # Tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')
    def train(self, datas):
        # collect infos for calculating word score: p(w|c)
        N = len(datas)
        classes   = [data['category'] for data in datas]
        documents = [data['content']  for data in datas]
        #
        self.document_count = N
        for i in range(N):
            c = classes[i]
            self.class_document_count[c] += 1
            if c not in self.class_set:
                self.class_set.add(c)
                self.class_count += 1
            #
            d = documents[i]
            self.class_total_word_count[c] += len(d)
            for w in d:
                self.word_class_count[w][c] += 1
                self.word_count[w] += 1 
                self.class_word_set[c].add(w)
            #
            # Print Progress
            if i%1000 == 0:
                print("Trained from", i, "documents")
            #
        # calculate word score: p(w|c)
        self.calc_word_score()
    def calc_word_score(self):
        '''
        word score: p(w|c), recorded in self.word_score[w][c]
        '''
        for w in self.word_class_count:
            # number of word w in each class
            class_count = self.word_class_count[w]
            # for each class
            for c in class_count:
                '''
                p(w|c) = (Wcw+1) / sum(Wcw+1))
                numerator: (Wcw+1)
                denominator: sum(Wcw+1)
                '''
                numerator = class_count[c] + 1
                denominator = self.class_total_word_count[c] + len(self.class_word_set[c])
                self.word_score[w][c] = numerator / denominator
    def predict(self, d):
        '''
        d is tokenized and stemmed
        use predict_after_preprocess() if input is raw string
        '''
        # p(c|d) = log(p(c)) + sum(log(p(w|c)))
        pcd = {} 
        # Get p(c|d) for each class
        for c in self.class_set:
            # log(p(c))
            pc = log(self.class_document_count[c] / self.document_count)
            # sum(log(p(w|c)))
            sum_ptc = sum([log(self.get_word_score(w, c)) for w in d])
            # p(c|d) = log(p(c)) + sum(log(p(w|c)))
            pcd[c] = pc + sum_ptc
        return (max(pcd, key=pcd.get), pcd)
    def predict_after_preprocess(self, content):
        '''
        content is raw string
        if tokenized and stemmed, use predict()
        '''
        d = self.document_preprocess(content)
        return self.predict(d)
    def document_preprocess(self, content):
        '''
        return tokenized and stemmed content
        '''
        tokenized_content = self.tokenizer.tokenize(content)
        stemmed_content   = [self.stemmer.stem(w) for w in tokenized_content]
        return stemmed_content
    def get_word_score(self, w, c):
        '''
        if word score is caluclated, return self.word_score[w][c]
        if not, return 1/(sum(Wcw+1))
        '''
        if w not in self.word_score or c not in self.word_score[w]:
            return 1 / (self.class_total_word_count[c] + len(self.class_word_set[c]))
            #return 1
        else:
            return self.word_score[w][c]

'''
Below is a sample of how to use this NaiveBayes
'''
nb = NaiveBayes()
import json
with open('datas/bbc_preproceseed.json') as fin:
    datas = json.load(fin)
nb.train(datas)
nb.predict_after_preprocess('apple and microsoft')
nb.predict(['ceremoni'])

predicts = []
for i in range(len(datas)):
    if (i%1000) == 0:
        print("Getting prediction of",i,'documents')
    predicts.append(nb.predict(datas[i]['title']))

pp = [p[0] for p in predicts]

correct = 0
for i in range(len(predicts)):
    if predicts[i][0] == datas[i]['category']:
        correct += 1

correct

