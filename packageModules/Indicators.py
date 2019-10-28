'''
Working...
This methods are independent of the nlp library and language.
Author: Eder Carbajo
'''
import re
import stanfordnlp
from cube.api import Cube
from nltk.tokenize import sent_tokenize, word_tokenize

class Indicators:

    def __init__(self, lang, structure):
        self.lang = lang
        self.structure = structure

    def calculate_num_words(self):
        sum = 0
        not_punctuation = lambda w: not (len(w) == 1 and (not w.isalpha()))
        for paragraph in self.structure:
            for sentence in paragraph:
                filterwords = filter(not_punctuation, word_tokenize(sentence.text))
                for word in filterwords:
                    sum = sum + 1
        return sum

    def calculate_num_paragraphs(self):
        return len(self.structure)

    def calculate_num_sentences(self):
        sum = 0
        for paragraph in self.structure:
            for sentence in paragraph:
                    sum = sum + 1
        return sum
