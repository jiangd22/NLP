import nltk
nltk.download('inaugural')
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import math
import string
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

        self.term_tf = []
        self.term_idf = {}



