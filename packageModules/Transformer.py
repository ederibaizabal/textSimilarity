import stanfordnlp
from cube.api import Cube
import re
from nltk import sent_tokenize

'''
This module transform a text into a unified data structure, which is going to be valid for every NLP libraries.
'''


class Processor:

    def __init__(self, lib):
        self.lib = lib
        self.text = None
        self.paragraph = []

    def process_text(self, text):
        self.text = text.replace('\n', '@')
        self.text = re.sub(r'@+', '@', self.text)
        # # print(self.text)
        # self.text = re.sub(r'[.]+(?![0-9])', r' . ', self.text)
        self.text = re.sub(r'[,]+(?![0-9])', r' , ', self.text)
        self.text = re.sub(r"!", " ! ", self.text)
        self.text = re.sub(r"\(", " ( ", self.text)
        self.text = re.sub(r"\)", " ) ", self.text)
        self.text = re.sub(r"\?", " ? ", self.text)
        self.text = re.sub(r";", " ; ", self.text)
        self.text = re.sub(r"\s{2,}", " ", self.text)
        print(self.text)
        return self.text


class ModelAdapter:

    def __init__(self, model, lib):
        self.model = model
        self.lib = lib

    def model_analysis(self, text):
        data = []
        if self.lib.lower() == "stanford":
            lines = text.split('@')
            for line in lines:  #paragraph
                paragraph = []
                if not line.strip() == '':
                    doc = self.model(line)
                    for sent in doc.sentences:
                        sequence = self.sent2sequenceStanford(sent)
                        print(sequence)
                        s = Sentence()
                        s.text = sequence
                        for word in sent.words:
                            #Por cada palabra de cada sentencia, creamos un objeto Word que contendra los attrs
                            w = Word()
                            w.index = str(word.index)
                            w.text = word.text
                            w.lemma = word.lemma
                            w.upos = word.upos
                            w.xpos = word.xpos
                            w.feats = word.feats
                            w.governor = word.governor
                            w.dependency_relation = word.dependency_relation
                            s.word_list.append(w)
                        paragraph.append(s)
                data.append(paragraph)

        elif self.lib.lower() == "cube":
            lines = text.split('@')
            for line in lines:
                paragraph = []
                sequences = self.model(line)
                for seq in sequences:
                    sequence = self.sent2sequenceCube(seq)
                    s = Sentence()
                    s.text = sequence
                    for entry in seq:
                        # Por cada palabra de cada sentencia, creamos un objeto Word que contendra los attrs
                        w = Word()
                        w.index = str(entry.index)
                        w.text = entry.word
                        w.lemma = entry.lemma
                        w.upos = entry.upos
                        w.xpos = entry.xpos
                        w.feats = entry.attrs
                        w.governor = str(entry.head)
                        w.dependency_relation = str(entry.label)
                        s.word_list.append(w)
                    paragraph.append(s)
                data.append(paragraph)
        # for p in data:
        #     for s in p:
        #         print(s.text)

        return data

    def sent2sequenceStanford(self, sent):
        conllword = ""
        for word in sent.words:
            conllword = conllword + " " + str(word.text)
        return conllword

    def sent2sequenceCube(self, sent):
        conllword = ""
        for entry in sent:
            conllword = conllword + " " + str(entry.word)
        return conllword
# class Paragraph:
#     sentence_list = []

class Sentence:
    word_list = []
    text = None

    def print(self):
        for words in Sentence.word_list:
            print(words.text)


class Word:
    index = None
    text = None
    lemma = None
    upos = None
    xpos = None
    feats = None
    governor = None
    dependency_relation = None
