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

    def __init__(self, lang, structure):
        self.lang = lang
        self.structure = structure
        self.indicators = defaultdict(float)
        self.aux_lists = defaultdict(list)
        self.words_freq = {}

    def calculate_num_words(self):
        self.indicators['num_words'] = 0
        not_punctuation = lambda w: not (len(w.text) == 1 and (not w.text.isalpha()))
        for paragraph in self.structure:
            self.aux_lists['sentences_per_paragraph'].append(len(paragraph))  # [1,2,1,...]
            for sentence in paragraph:
                filterwords = filter(not_punctuation, sentence.word_list)
                sum = 0
                for word in filterwords:
                    self.indicators['num_words'] += 1
                    self.aux_lists['words_length_list'].append(len(word.text))
                    self.aux_lists['lemmas_length_list'].append(len(word.lemma))
                    sum += 1
                self.aux_lists['sentences_length_mean'].append(sum)

    def calculate_num_paragraphs(self):
        self.indicators['num_paragraphs'] = len(self.structure)

    def calculate_num_sentences(self):
        self.indicators['num_sentences'] = 0
        for paragraph in self.structure:
            for sentence in paragraph:
                self.indicators['num_sentences'] += 1

    def calculate_all_means(self):
        i = self.indicators
        i['sentences_per_paragraph_mean'] = round(float(np.mean(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_mean'] = round(float(np.mean(self.aux_lists['sentences_length_mean'])), 4)
        i['words_length_mean'] = round(float(np.mean(self.aux_lists['words_length_list'])), 4)
        i['lemmas_length_mean'] = round(float(np.mean(self.aux_lists['lemmas_length_list'])), 4)

    def calculate_all_std_deviations(self):
        i = self.indicators
        i['sentences_per_paragraph_std'] = round(float(np.std(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_std'] = round(float(np.std(self.aux_lists['sentences_length_mean'])), 4)
        i['words_length_std'] = round(float(np.std(self.aux_lists['words_length_list'])), 4)
        i['lemmas_length_std'] = round(float(np.std(self.aux_lists['lemmas_length_list'])), 4)

    def analyze(self):
        self.calculate_num_words()
        self.calculate_num_paragraphs()
        self.calculate_num_sentences()
        self.calculate_all_means()
        self.calculate_all_std_deviations()
        return self.indicators

