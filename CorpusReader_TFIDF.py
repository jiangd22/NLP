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

    # tfidf(fileid, returnZero = false) : return the TF-IDF for the specific document in the corpus (specified by fileid). The vector is represented by a dictionary/hash in python. The keys are the terms, and the values are the tf-idf value of the dimension. If returnZero is true, then the dictionary will contain terms that have 0 value for that vector, otherwise the vector will omit those terms
    def tfidf(self, fileid, returnZero = False):
        words = self.wordProcess(self.words(fileid))
        tfidf_dict = defaultdict(float)
        tf_dict = defaultdict(float)
        if (self.tf_method == "raw"):
            for word in words:
                tf_dict[word] += 1
            for word in tf_dict:
                tf_dict[word] = tf_dict[word] / len(words)
        elif (self.tf_method == "log"):
            for word in words:
                tf_dict[word] += 1
            for word in tf_dict:
                tf_dict[word] = 1 + math.log2(tf_dict[word]/len(words))

        N = len(self.corpus.fileids())
        idf_dict = defaultdict(float)
        for word in words:
            ni = 0
            for fileid in self.corpus.fileids():
                if word in self.wordProcess(self.words(fileid)):
                    ni += 1
            if self.idf_method == "base":
                idf_dict[word] = math.log2(N / ni)
            elif self.idf_method == "smooth":
                idf_dict[word] = math.log2(1+(N / ni))

        for word in words:
            if returnZero:
                tfidf_dict[word] = tf_dict[word] * idf_dict[word]
            else:
                if tf_dict[word] * idf_dict[word] != 0:
                    tfidf_dict[word] = tf_dict[word] * idf_dict[word]

        return tfidf_dict

    # tfidfAll(returnZero = false) : return the TF-IDF for all documents in the corpus. It will be returned as a dictionary. The key is the fileid of each document, for each document the value is the tfidf of that document (using the same format as above).
    def tfidfAll(self, returnZero = False):
        tfidfAll_dict = defaultdict(float)
        for fileid in self.corpus.fileids():
            tfidfAll_dict[fileid] = self.tfidf(fileid, returnZero)
        return tfidfAll_dict

    #tfidfNew([words]) : return the tf-idf of a “new” document, represented by a list of words. You should honor the various parameters (ignoreCase, toStem etc.) when preprocessing the new document. Also, the idf of each word should not be changed (i.e. the “new” document should not be treated as part of the corpus).tfidfNew([words]) : return the tf-idf of a “new” document, represented by a list of words. You should honor the various parameters (ignoreCase, toStem etc.) when preprocessing the new document. Also, the idf of each word should not be changed (i.e. the “new” document should not be treated as part of the corpus).
    def tfidfNew(self, words):
        words = self.wordProcess(words)
        tfidf_dict = {}
        for word in words:
            tfidf_dict[word] += 1
        for word in tfidf_dict:
            tfidf_dict[word] = tfidf_dict[word] / len(words)
        return tfidf_dict

    # idf() : return the idf of each term as a dictionary : keys are the terms, and values are the idf
    def idf(self):
        idf_dict = {}
        for fileid in self.corpus.fileids():
            words = self.wordProcess(self.words(fileid))
            for word in words:
                idf_dict[word] += 1
        for word in idf_dict:
            idf_dict[word] = math.log(len(self.corpus.fileids()) / idf_dict[word])
        return idf_dict

    # cosine_sim([fileid1, fileid2]) return the cosine similarity between two documents in the corpus
    def cosine_sim(self, fileid1, fileid2):
        tfidf_dict = self.tfidfAll()
        words1 = tfidf_dict[fileid1]
        words2 = tfidf_dict[fileid2]
        words = set(words1.keys()) | set(words2.keys())
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        for word in words:
            numerator += words1[word] * words2[word]
            denominator1 += words1[word] ** 2
            denominator2 += words2[word] ** 2
        return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))

    # cosine_sim_new([words], fileid): return the cosine similary between a “new” document (as if
    # specified like the tfidf_new() method) and the documents specified by fileid.
    def cosine_sim_new(self, words, fileid):
        tfidf_dict = self.tfidfAll()
        words1 = tfidf_dict[fileid]
        words2 = self.tfidfNew(words)
        words = set(words1.keys()) | set(words2.keys())
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        for word in words:
            numerator += words1[word] * words2[word]
            denominator1 += words1[word] ** 2
            denominator2 += words2[word] ** 2
        return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))

    # query([words]) : return a list of (document, cosine_sim) tuples that calculate the cosine similarity between the “new” document (specified by the list of words as the document). The list should be ordered in decreasing order of cosine similarity.
    def query(self, words):
        tfidf_dict = self.tfidfAll()
        words2 = self.tfidfNew(words)
        words = set(words2.keys())
        query_list = []
        for fileid in self.corpus.fileids():
            words1 = tfidf_dict[fileid]
            numerator = 0
            denominator1 = 0
            denominator2 = 0
            for word in words:
                numerator += words1[word] * words2[word]
                denominator1 += words1[word] ** 2
                denominator2 += words2[word] ** 2
            query_list.append((fileid, numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))))
        query_list.sort(key=lambda x: x[1], reverse=True)
        return query_list