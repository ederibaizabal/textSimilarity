import numpy as np
from collections import defaultdict

'''
This module transform a text into a unified data structure, which is going to be valid for every NLP libraries.
'''


class ModelAdapter:

    def __init__(self, model, lib):
        # parser
        self.model = model
        # model_name
        self.lib = lib

    def model_analysis(self, text):
        # Carga la estructura unificada con el analisis del parser,
        # [[sentences],[sentences],[sentences]...]
        d = Document(text)
        if self.lib.lower() == "stanford":
            lines = text.split('@')
            for line in lines:  # paragraph
                p = Paragraph()
                if not line.strip() == '':
                    doc = self.model(line)
                    for sent in doc.sentences:
                        s = Sentence()
                        sequence = self.sent2sequenceStanford(sent)
                        s.text = sequence
                        for word in sent.words:
                            # Por cada palabra de cada sentencia, creamos un objeto Word que contendra los attrs
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
                            print(str(
                                word.index) + "\t" + word.text + "\t" + word.lemma + "\t" + word.upos + "\t" + word.xpos + "\t" + word.feats + "\t")
                        p.sentence_list.append(s)
                    d.paragraph_list.append(p)

        elif self.lib.lower() == "cube":
            lines = text.split('@')
            for line in lines:
                p = Paragraph()
                sequences = self.model(line)
                for seq in sequences:
                    s = Sentence()
                    sequence = self.sent2sequenceCube(seq)
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
                        print(str(
                            w.index) + "\t" + w.text + "\t" + w.lemma + "\t" + w.upos + "\t" + w.xpos + "\t" + w.feats + "\t")
                        s.word_list.append(w)
                    p.sentence_list.append(s)
                d.paragraph_list.append(p)
        return d

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


class Document:
    def __init__(self, text):
        self._text = text
        self._paragraph_list = []
        # Indicadores
        self.indicators = defaultdict(float)
        self.aux_lists = defaultdict(list)

    @property
    def text(self):
        """ Access text of this document. Example: 'This is a sentence.'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the document's text value. Example: 'This is a sentence.'"""
        self._text = value

    @property
    def paragraph_list(self):
        """ Access list of sentences for this document. """
        return self._paragraph_list

    @paragraph_list.setter
    def paragraph_list(self, value):
        """ Set the list of tokens for this document. """
        self._paragraph_list = value

    def get_indicators(self):
        self.indicators['num_sentences'] = self.calculate_num_sentences()
        self.indicators['num_words'] = self.calculate_num_words()
        self.indicators['num_paragraphs'] = self.calculate_num_paragraphs()
        self.analyze_iterator()
        self.calculate_all_means()
        self.calculate_all_std_deviations()
        self.calculate_all_incidence()
        return self.indicators

    def calculate_num_words(self):
        num_words = 0
        not_punctuation = lambda w: not (len(w.text) == 1 and (not w.text.isalpha()))
        for paragraph in self._paragraph_list:
            self.aux_lists['sentences_per_paragraph'].append(len(paragraph.sentence_list))  # [1,2,1,...]
            for sentence in paragraph.sentence_list:
                filterwords = filter(not_punctuation, sentence.word_list)
                sum = 0
                for word in filterwords:
                    num_words += 1
                    self.aux_lists['words_length_list'].append(len(word.text))
                    self.aux_lists['lemmas_length_list'].append(len(word.lemma))
                    sum += 1
                self.aux_lists['sentences_length_mean'].append(sum)
        return num_words

    def calculate_num_paragraphs(self):
        return len(self._paragraph_list)

    def calculate_num_sentences(self):
        num_sentences = 0
        for paragraph in self._paragraph_list:
            for sentence in paragraph.sentence_list:
                num_sentences += 1
        return num_sentences

    def analyze_iterator(self):
        i = self.indicators
        for p in self.paragraph_list:
            for s in p.sentence_list:
                for w in s.word_list:
                    atributos = w.feats.split('|')
                    if 'Mood=Imp' in atributos:
                        i['num_impera'] += 1
                    if 'PronType=Prs' in atributos:
                        i['num_personal_pronouns'] += 1
                        if 'Person=1' in atributos:
                            i['num_first_pers_pron'] += 1
                            if 'Number=Sing' in atributos:
                                i['num_first_pers_sing_pron'] += 1
                        elif 'Person=3' in atributos:
                            i['num_third_pers_pron'] += 1

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

    @staticmethod
    def get_incidence(indicador, num_words):
        return round(((1000 * indicador) / num_words), 4)

    def calculate_all_incidence(self):
        i = self.indicators
        i['num_impera_incidence'] = self.get_incidence(i['num_impera'], i['num_words'])
        i['num_personal_pronouns_incidence'] = self.get_incidence(i['num_personal_pronouns'], i['num_words'])
        i['num_first_pers_pron_incidence'] = self.get_incidence(i['num_first_pers_pron'], i['num_words'])
        i['num_first_pers_sing_pron_incidence'] = self.get_incidence(i['num_first_pers_sing_pron'], i['num_words'])
        i['num_third_pers_pron_incidence'] = self.get_incidence(i['num_third_pers_pron'], i['num_words'])


class Paragraph:

    def __init__(self):
        self._sentence_list = []

    @property
    def sentence_list(self):
        """ Access list of sentences for this document. """
        return self._sentence_list

    @sentence_list.setter
    def sentence_list(self, value):
        """ Set the list of tokens for this document. """
        self.sentence_list = value


class Sentence:

    def __init__(self):
        self._word_list = []
        self.text = None

    @property
    def word_list(self):
        """ Access list of words for this sentence. """
        return self._word_list

    @word_list.setter
    def word_list(self, value):
        """ Set the list of words for this sentence. """
        self._word_list = value

    def print(self):
        for words in self.word_list:
            print(words.text)


class Word:
    def __init__(self):
        self._index = None
        self._text = None
        self._lemma = None
        self._upos = None
        self._xpos = None
        self._feats = None
        self._governor = None
        self._dependency_relation = None

    @property
    def dependency_relation(self):
        """ Access dependency relation of this word. Example: 'nmod'"""
        return self._dependency_relation

    @dependency_relation.setter
    def dependency_relation(self, value):
        """ Set the word's dependency relation value. Example: 'nmod'"""
        self._dependency_relation = value

    @property
    def lemma(self):
        """ Access lemma of this word. """
        return self._lemma

    @lemma.setter
    def lemma(self, value):
        """ Set the word's lemma value. """
        self._lemma = value

    @property
    def governor(self):
        """ Access governor of this word. """
        return self._governor

    @governor.setter
    def governor(self, value):
        """ Set the word's governor value. """
        self._governor = value

    @property
    def pos(self):
        """ Access (treebank-specific) part-of-speech of this word. Example: 'NNP'"""
        return self._xpos

    @pos.setter
    def pos(self, value):
        """ Set the word's (treebank-specific) part-of-speech value. Example: 'NNP'"""
        self._xpos = value

    @property
    def text(self):
        """ Access text of this word. Example: 'The'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the word's text value. Example: 'The'"""
        self._text = value

    @property
    def xpos(self):
        """ Access treebank-specific part-of-speech of this word. Example: 'NNP'"""
        return self._xpos

    @xpos.setter
    def xpos(self, value):
        """ Set the word's treebank-specific part-of-speech value. Example: 'NNP'"""
        self._xpos = value

    @property
    def upos(self):
        """ Access universal part-of-speech of this word. Example: 'DET'"""
        return self._upos

    @upos.setter
    def upos(self, value):
        """ Set the word's universal part-of-speech value. Example: 'DET'"""
        self._upos = value

    @property
    def feats(self):
        """ Access morphological features of this word. Example: 'Gender=Fem'"""
        return self._feats

    @feats.setter
    def feats(self, value):
        """ Set this word's morphological features. Example: 'Gender=Fem'"""
        self._feats = value

    @property
    def parent_token(self):
        """ Access the parent token of this word. """
        return self._parent_token

    @parent_token.setter
    def parent_token(self, value):
        """ Set this word's parent token. """
        self._parent_token = value

    @property
    def index(self):
        """ Access index of this word. """
        return self._index

    @index.setter
    def index(self, value):
        """ Set the word's index value. """
        self._index = value

    # def is_lexic_word(self, entry, sequence):
    #     return self.is_verb(entry, sequence) or entry.upos == 'NOUN' or entry.upos == 'ADJ' or entry.upos == 'ADV'
    #
    # def is_verb(self, word, sequence):
    #     return word.upos == 'VERB' or (word.upos == 'AUX' and sequence[word.governor - 1].upos != 'VERB')

    def __repr__(self):
        features = ['index', 'text', 'lemma', 'upos', 'xpos', 'feats', 'governor', 'dependency_relation']
        feature_str = ";".join(["{}={}".format(k, getattr(self, k)) for k in features if getattr(self, k) is not None])

        return f"<{self.__class__.__name__} {feature_str}>"
