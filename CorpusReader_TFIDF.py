import nltk
nltk.download('inaugural')
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import math
import string
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer

class CorpusReader_TFIDF:
    # pass
    def __init__(self, corpus, tf="raw", idf="base", stopWord="none", toStem=False, stemFirst=False, ignoreCase=True):
        self.corpus = corpus
        self.tf_method = tf
        self.idf_method = idf
        self.stopWord_method = stopWord
        self.toStem = toStem
        self.stemFirst = stemFirst
        self.ignoreCase = ignoreCase

        # self.term_tf = []
        # self.term_idf = {}
        # self.term_tfidf = {}

    def fields(self):
        return self.corpus.fields()

    def raw(self):
        return self.corpus.raw()

    def raw(self, fileid = None):
        return self.corpus.raw(fileid)

    def words(self):
        return self.corpus.words()

    def words(self, fileid = None):
        return self.corpus.words(fileid)

    def wordProcess(self, words):
        if self.ignoreCase:
            words = [word.lower() for word in words]
        if self.toStem:
            if self.stemFirst:
                stemmer = SnowballStemmer("english")
                words = [stemmer.stem(word) for word in words]
                if self.stopWord_method == "none" or self.stopWord_method == None:
                    pass
                elif self.stopWord_method == "standard":
                    stop_words = set(nltk.corpus.stopwords.words('english'))
                    words = [word for word in words if word not in stop_words]
                else:
                    with open(self.stopWord_method, 'r') as SWord_Bank:
                        stop_words = set(line.strip() for line in SWord_Bank)
                    words = [word for word in words if word not in stop_words]
            else:
                if self.stopWord_method == "none" or self.stopWord_method == None:
                    pass
                elif self.stopWord_method == "standard":
                    stop_words = set(nltk.corpus.stopwords.words('english'))
                    words = [word for word in words if word not in stop_words]
                else:
                    with open(self.stopWord_method, 'r') as SWord_Bank:
                        stop_words = set(line.strip() for line in SWord_Bank)
                    words = [word for word in words if word not in stop_words]
                stemmer = SnowballStemmer("english")
                words = [stemmer.stem(word) for word in words]
        return words

    #tfidf(fileid, returnZero = false) : return the TF-IDF for the specific document in the corpus (specified by fileid). The vector is represented by a dictionary/hash in python. The keys are the terms, and the values are the tf-idf value of the dimension. If returnZero is true, then the dictionary will contain terms that have 0 value for that vector, otherwise the vector will omit those terms
    def tfidf(self, fileid, returnZero = False):
        tfidf_dict = {}
        tf = self.tf(fileid)
        idf = self.idf()
        for term in tf:
            tfidf_dict[term] = tf[term] * idf[term]
        if returnZero:
            return tfidf_dict
        else:
            return {k: v for k, v in tfidf_dict.items() if v != 0}






