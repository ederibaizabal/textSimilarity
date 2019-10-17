import stanfordnlp
from cube.api import Cube
import re
from nltk import sent_tokenize

'''
This module transform a text into a unified data structure, which is going to be valid for every NLP libraries.
'''

class Transformer:

    def __init__(self, text, lib):
        self.text = text
        self.lib = lib
        self.paragraph = []

    def get_paragraph(self):
        lines = self.text.split("\n\n")
        # paragraph [ [sentences, words], [sentences, words], ...]
        for line in lines:
            paraData = []
            paraData.append(self.get_sentences(line))       #adding the sentences of the paragraph
            paraData.append(self.get_words(line))           #adding the words of the paragraph
            self.paragraph.append(paraData)
        return self.paragraph

    def get_sentences(self, sentence):
        return sent_tokenize(sentence)

    def get_words(self, line):
        res = re.findall(r'\w+', line)
        #print(str(res))
        return res


