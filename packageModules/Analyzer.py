'''
Working...
This methods are independent of the nlp library and language.
Author: Eder Carbajo
'''
import re
import stanfordnlp
from cube.api import Cube
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import numpy as np


class Analyzer:
    indicators = defaultdict(float)

    def __init__(self):
        self.aux_lists = defaultdict(list)
        self.words_freq = {}

    def analyze(self):
        self.calculate_num_words()
        self.calculate_num_paragraphs()
        self.calculate_num_sentences()
        self.calculate_all_means()
        self.calculate_all_std_deviations()
        return self.indicators

