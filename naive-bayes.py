from collections import defaultdict
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from math import log

class NaiveBayes:
    def __init__(self):
        # total number of total documents
        self.document_count       = 0
        # all possible classes
        self.class_set            = set()
        # total number of classes
        self.class_count          = 0
        # number of documents in each classes
        self.class_document_count = defaultdict(lambda: 0)
        # number of each words in each classes [word][category]
        self.word_class_count     = defaultdict(lambda: defaultdict(lambda: 0))
        # total number of each words
        self.word_count           = defaultdict(lambda: 0)
        # word score = P(t|C), set default value to eliminate zeros
        self.word_score           = defaultdict(lambda: defaultdict(lambda: 1))
        #
        # Stemmer
        self.stemmer = PorterStemmer()
        # Tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')
    def train(self, datas):
        N = len(datas)
        classes =   [data['category'] for data in datas]
        documents = [data['content']    for data in datas]
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
            for w in d:
                self.word_class_count[w][c] += 1
                self.word_count[w] += 1 
            #
            # Print Progress
            if i%1000 == 0:
                print("Processed", i, "documents")
            #
        self.calc_word_score()
    def calc_word_score(self):
        for w in self.word_class_count:
            class_count = self.word_class_count[w]
            denominator = sum(class_count.values()) + len(class_count)
            for c in class_count:
                numerator = class_count[c] + 1
                self.word_score[w][c] = numerator / denominator
    def predict(self, content):
        d = self.document_preprocess(content)
        pcd = {}
        for c in self.class_set:
            pc = log(self.class_document_count[c] / self.document_count)
            sum_ptc = sum([log(self.word_score[w][c]) for w in content])
            pcd[c] = pc + sum_ptc
        return (max(pcd, key=pcd.get), pcd)
    def document_preprocess(self, content):
        tokenized_content = self.tokenizer.tokenize(content)
        stemmed_content   = [self.stemmer.stem(w) for w in tokenized_content]
        return stemmed_content

nb = NaiveBayes()
nb.train(datas)
nb.predict('apple and microsoft')
