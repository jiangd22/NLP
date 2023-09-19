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
    # CorpusReader_TFIDF Constructor.
    def __init__(self, corpus, tf="raw", idf="base", stopWord="none", toStem=False, stemFirst=False, ignoreCase=True):
        self.corpus = corpus
        self.tf_method = tf
        self.idf_method = idf
        self.stopWord_method = stopWord
        self.toStem = toStem
        self.stemFirst = stemFirst
        self.ignoreCase = ignoreCase

    # Shared Methods in the corpus reader class.
    def fields(self):
        return self.corpus.fields()

    def raw(self):
        return self.corpus.raw()

    def raw(self, fileid=None):
        return self.corpus.raw(fileid)

    def words(self):
        return self.corpus.words()

    def words(self, fileid=None):
        return self.corpus.words(fileid)

    # Word processing methods for the corpus reader class. So stemming, stopword removal, lowercasing, etc.
    # E.g. toStem, ignoreCase, stemFirst, stopWord
    def wordProcess(self, words):
        if self.ignoreCase:
            words = [word.lower() for word in words]
        if self.toStem:
            if self.stemFirst:
                stemmer = SnowballStemmer("english")
                words = [stemmer.stem(word) for word in words]
                if self.stopWord_method == "none" or self.stopWord_method is None:
                    pass
                elif self.stopWord_method == "standard":
                    stop_words = set(nltk.corpus.stopwords.words('english'))
                    words = [word for word in words if word not in stop_words]
                else:
                    with open(self.stopWord_method, 'r') as SWord_Bank:
                        stop_words = set(line.strip() for line in SWord_Bank)
                    words = [word for word in words if word not in stop_words]
            else:
                if self.stopWord_method == "none" or self.stopWord_method is None:
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

    # Methods specific to CorpusReader_TFIDF class. E.g. tfidf, cosine_sim, etc.
    def tfidf(self, fileid):
        words = self.wordProcess(self.words(fileid))
        tfidf_dict = defaultdict(float)
        tf_dict = defaultdict(float)
        if self.tf_method == "raw":
            for word in words:
                tf_dict[word] += 1
            for word in tf_dict:
                tf_dict[word] = tf_dict[word] / len(words)
        elif self.tf_method == "log":
            for word in words:
                tf_dict[word] += 1
            for word in tf_dict:
                tf_dict[word] = 1 + math.log2(tf_dict[word] / len(words))

        # N = len(self.corpus.fileids())
        # idf_dict = defaultdict(float)
        # for word in words:
        #     ni = 0
        #     for fileid in self.corpus.fileids():
        #         if word in self.wordProcess(self.words(fileid)):
        #             ni += 1
        #     if self.idf_method == "base":
        #         idf_dict[word] = math.log2(N / ni)
        #     elif self.idf_method == "smooth":
        #         idf_dict[word] = math.log2(1+(N / ni))

        for word in words:
            tfidf_dict[word] = tf_dict[word] * self.idf()[word]

        return tfidf_dict
    # This is just the overloaded tfidf method with the returnZero parameter. It's used in the tfidfAll method.
    def tfidf(self, fileid, returnZero=False):
        print("tfidf start")
        words = self.wordProcess(self.words(fileid))
        tfidf_dict = defaultdict(float)
        tf_dict = defaultdict(float)
        if self.tf_method == "raw":
            print("tfidf #1")
            for word in words:
                tf_dict[word] += 1
            for word in tf_dict:
                tf_dict[word] = tf_dict[word] / len(words)
        elif self.tf_method == "log":
            print("tfidf #1")
            for word in words:
                tf_dict[word] += 1
            for word in tf_dict:
                tf_dict[word] = 1 + math.log2(tf_dict[word] / len(words))

        N = len(self.corpus.fileids())
        idf_dict = defaultdict(float)
        for word in self.words():
            ni = 0
            for fileid in self.corpus.fileids():
                if word in self.wordProcess(self.words(fileid)):
                    ni += 1
            if self.idf_method == "base":
                if ni == 0:
                    idf_dict[word] = 0
                else:
                    idf_dict[word] = math.log2(N / ni)
            elif self.idf_method == "smooth":
                idf_dict[word] = math.log2(1 + (N / ni))
        print("tfidf #2")
        for word in words:
            print("tfidf #2.1")
            print(tf_dict[word])
            print(idf_dict[word])
            if returnZero:
                print("tfidf #2.2")
                tfidf_dict[word] = tf_dict[word] * idf_dict[word]
            else:
                tfidf_dict[word] = tf_dict[word] * idf_dict[word]
        return tfidf_dict


    def tfidfAll(self, returnZero=False):
        tfidfAll_dict = defaultdict(float)
        for fileid in self.corpus.fileids():
            tfidfAll_dict[fileid] = self.tfidf(fileid, returnZero)[fileid]
        return tfidfAll_dict

    def tfidfNew(self, words):
        word = self.wordProcess(words)
        tfidfNew_dict = defaultdict(float)
        for word in words:
            tfidfNew_dict[word] = self.tfidf(word)[word]
        return tfidfNew_dict

    def idf(self):
        N = len(self.corpus.fileids())
        idf_dict = defaultdict(float)
        for word in self.words():
            ni = 0
            for fileid in self.corpus.fileids():
                if word in self.wordProcess(self.words(fileid)):
                    ni += 1
            if self.idf_method == "base":
                if ni == 0:
                    idf_dict[word] = 0
                else:
                    idf_dict[word] = math.log2(N / ni)
            elif self.idf_method == "smooth":
                idf_dict[word] = math.log2(1 + (N / ni))
        return idf_dict

    def cosine_sim(self, fileid1, fileid2):
        file1 = self.tfidf(fileid1)
        file2 = self.tfidf(fileid2)
        words = set(file1.keys()) | set(file2.keys())
        vector1 = [file1[word] for word in words]
        vector2 = [file2[word] for word in words]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def cosine_sim_new(self, words, fileid):
        file1 = self.tfidfNew(words)
        file2 = self.tfidf(fileid)
        words = set(file1.keys()) | set(file2.keys())
        vector1 = [file1[word] for word in words]
        vector2 = [file2[word] for word in words]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))




