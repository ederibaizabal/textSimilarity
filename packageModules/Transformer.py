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
                                w.index) + "\t" + w.text + "\t" + w.lemma + "\t" + w.upos + "\t" +
                                  w.xpos + "\t" + w.feats + "\t" + str(w.governor) + "\t" + str(w.dependency_relation) +
                                  "\t")
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
        self.words_freq = {}
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
        self.analyze()
        self.calculate_all_means()
        self.calculate_all_std_deviations()
        self.calculate_all_incidence()
        self.calculate_density()
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

    def calculate_left_embeddedness(self, sequences):
        list_left_embeddedness = []
        for sequence in sequences:
            verb_index = 0
            main_verb_found = False
            left_embeddedness = 0
            num_words = 0
            for word in sequence.word_list:
                if not len(word.text) == 1 or word.text.isalpha():
                    if not main_verb_found and word.governor < len(sequence.word_list):
                        if self.is_verb(word, sequence):
                            verb_index += 1
                            if (word.upos == 'VERB' and word.dependency_relation == 'root') or (
                                    word.upos == 'AUX' and sequence.word_list[word.governor].dependency_relation == 'root'
                                    and sequence.word_list[word.governor].upos == 'VERB'):
                                main_verb_found = True
                                left_embeddedness = num_words
                            if verb_index == 1:
                                left_embeddedness = num_words
                    num_words += 1
            list_left_embeddedness.append(left_embeddedness)
        self.indicators['left_embeddedness'] = round(float(np.mean(list_left_embeddedness)), 4)

    def count_np_in_sentence(self, sentence):
        list_np_indexes = []
        for word in sentence.word_list:
            if word.upos == 'NOUN' or word.upos == 'PRON' or word.upos == 'PROPN':
                if word.dependency_relation in ['fixed', 'flat', 'compound']:
                    if word.governor not in list_np_indexes:
                        list_np_indexes.append(word.governor)
                else:
                    if word.index not in list_np_indexes:
                        ind = int(word.index)
                        list_np_indexes.append(ind)
        return list_np_indexes

    def is_verb(self, word, sequence):
        return word.upos == 'VERB' or (word.upos == 'AUX' and sequence.word_list[word.governor - 1].upos != 'VERB')

    def is_lexic_word(self, entry, sequence):
        return self.is_verb(entry, sequence) or entry.upos == 'NOUN' or entry.upos == 'ADJ' or entry.upos == 'ADV'

    def count_modifiers(self, sentence, list_np_indexes):
        num_modifiers_per_np = []
        for index in list_np_indexes:
            num_modifiers = 0
            for entry in sentence.word_list:
                if int(entry.governor) == int(index) and entry.has_modifier():
                    num_modifiers += 1
            num_modifiers_per_np.append(num_modifiers)
        return num_modifiers_per_np

    def count_decendents(self, sentence, list_np_indexes):
        num_modifiers = 0
        if len(list_np_indexes) == 0:
            return num_modifiers
        else:
            new_list_indexes = []
            for entry in sentence.word_list:
                if entry.governor in list_np_indexes and entry.has_modifier():
                    new_list_indexes.append(entry.index)
                    num_modifiers += 1
            return num_modifiers + self.count_decendents(sentence, new_list_indexes)

    def count_vp_in_sentence(self, sentence):
        num_np = 0
        for entry in sentence.word_list:
            if self.is_verb(entry, sentence):
                num_np += 1
        return num_np

    def get_num_hapax_legomena(self):
        num_hapax_legonema = 0
        for word, frecuencia in self.words_freq.items():
            if frecuencia == 1:
                num_hapax_legonema += 1
        return num_hapax_legonema

    def calculate_honore(self):
        n = self.indicators['num_words']
        v = len(self.aux_lists['different_forms'])
        v1 = self.get_num_hapax_legomena()
        self.indicators['honore'] = round(100 * ((np.log10(n)) / (1 - (v1 / v))), 4)

    def calculate_maas(self):
        n = self.indicators['num_words']
        v = len(self.aux_lists['different_forms'])
        self.indicators['maas'] = round((np.log10(n) - np.log10(v)) / (np.log10(v) ** 2), 4)

    def analyze(self):
        i = self.indicators
        num_np_list = []
        num_vp_list = []
        modifiers_per_np = []
        subordinadas_labels = ['csubj', 'csubj:pass', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl']
        decendents_total = 0

        for p in self.paragraph_list:
            self.calculate_left_embeddedness(p.sentence_list)
            for s in p.sentence_list:
                vp_indexes = self.count_np_in_sentence(s)
                num_np_list.append(len(vp_indexes))
                num_vp_list.append(self.count_vp_in_sentence(s))
                decendents_total += self.count_decendents(s, vp_indexes)
                modifiers_per_np += self.count_modifiers(s, vp_indexes)
                i['prop'] = 0
                numPunct = 0
                for w in s.word_list:
                    if self.is_lexic_word(w, s):
                        i['num_lexic_words'] += 1
                    if w.upos == 'NOUN':
                        i['num_noun'] += 1
                    if w.upos == 'ADJ':
                        i['num_adj'] += 1
                    if w.upos == 'ADV':
                        i['num_adv'] += 1
                    if self.is_verb(w, s):
                        i['num_verb'] += 1
                    if w.text.lower() not in self.aux_lists['different_forms']:
                        self.aux_lists['different_forms'].append(w.text.lower())
                    if w.text.lower() not in self.words_freq:
                        self.words_freq[w.text.lower()] = 1
                    else:
                        self.words_freq[w.text.lower()] = self.words_freq.get(w.text.lower()) + 1
                    if w.dependency_relation in subordinadas_labels:
                        i['num_subord'] += 1
                        # Numero de sentencias subordinadas relativas
                        if w.dependency_relation == 'acl:relcl':
                            i['num_rel_subord'] += 1
                    if w.upos == 'PUNCT':
                        numPunct += 1
                    if w.dependency_relation == 'conj' or w.dependency_relation == 'csubj' or w.dependency_relation == 'csubj:pass' or w.dependency_relation == 'ccomp' or w.dependency_relation == 'xcomp' or w.dependency_relation == 'advcl' or w.dependency_relation == 'acl' or w.dependency_relation == 'acl:relcl':
                        i['prop'] += 1
                    atributos = w.feats.split('|')
                    if 'VerbForm=Ger' in atributos:
                        i['num_ger'] += 1
                    if 'VerbForm=Inf' in atributos:
                        i['num_inf'] += 1
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
                i['num_total_prop'] = i['num_total_prop'] + i['prop']
                self.aux_lists['prop_per_sentence'].append(i['prop'])
                self.aux_lists['punct_per_sentence'].append(numPunct)

        i['num_different_forms'] = len(self.aux_lists['different_forms'])
        self.calculate_honore()
        self.calculate_maas()
        i['num_decendents_noun_phrase'] = round(decendents_total / sum(num_np_list), 4)
        i['num_modifiers_noun_phrase'] = round(float(np.mean(modifiers_per_np)), 4)
        self.calculate_phrases(num_vp_list, num_np_list)

    def calculate_all_means(self):
        i = self.indicators
        i['sentences_per_paragraph_mean'] = round(float(np.mean(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_mean'] = round(float(np.mean(self.aux_lists['sentences_length_mean'])), 4)
        i['words_length_mean'] = round(float(np.mean(self.aux_lists['words_length_list'])), 4)
        i['lemmas_length_mean'] = round(float(np.mean(self.aux_lists['lemmas_length_list'])), 4)
        i['mean_propositions_per_sentence'] = round(float(np.mean(self.aux_lists['prop_per_sentence'])), 4)
        i['num_punct_marks_per_sentence'] = round(float(np.mean(self.aux_lists['punct_per_sentence'])), 4)

    def calculate_all_std_deviations(self):
        i = self.indicators
        i['sentences_per_paragraph_std'] = round(float(np.std(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_std'] = round(float(np.std(self.aux_lists['sentences_length_mean'])), 4)
        i['words_length_std'] = round(float(np.std(self.aux_lists['words_length_list'])), 4)
        i['lemmas_length_std'] = round(float(np.std(self.aux_lists['lemmas_length_list'])), 4)

    def calculate_phrases(self, num_vp_list, num_np_list):
        i = self.indicators
        i['mean_vp_per_sentence'] = round(float(np.mean(num_vp_list)), 4)
        i['mean_np_per_sentence'] = round(float(np.mean(num_np_list)), 4)
        i['noun_phrase_density_incidence'] = self.get_incidence(sum(num_np_list), i['num_words'])
        i['verb_phrase_density_incidence'] = self.get_incidence(sum(num_vp_list), i['num_words'])

    @staticmethod
    def get_incidence(indicador, num_words):
        return round(((1000 * indicador) / num_words), 4)

    def calculate_all_incidence(self):
        i = self.indicators
        n = i['num_words']
        i['num_sentences_incidence'] = self.get_incidence(i['num_sentences'], n)
        i['num_paragraphs_incidence'] = self.get_incidence(i['num_paragraphs'], n)
        i['num_impera_incidence'] = self.get_incidence(i['num_impera'], n)
        i['num_personal_pronouns_incidence'] = self.get_incidence(i['num_personal_pronouns'], n)
        i['num_first_pers_pron_incidence'] = self.get_incidence(i['num_first_pers_pron'], n)
        i['num_first_pers_sing_pron_incidence'] = self.get_incidence(i['num_first_pers_sing_pron'], n)
        i['num_third_pers_pron_incidence'] = self.get_incidence(i['num_third_pers_pron'], n)
        i['gerund_density_incidence'] = self.get_incidence(i['num_ger'], n)
        i['infinitive_density_incidence'] = self.get_incidence(i['num_inf'], n)
        i['num_subord_incidence'] = self.get_incidence(i['num_subord'], n)
        i['num_rel_subord_incidence'] = self.get_incidence(i['num_rel_subord'], n)

    def calculate_density(self):
        i = self.indicators
        i['lexical_density'] = round(i['num_lexic_words'] / i['num_words'], 4)
        i['noun_density'] = round(i['num_noun'] / i['num_words'], 4)
        i['verb_density'] = round(i['num_verb'] / i['num_words'], 4)
        i['adj_density'] = round(i['num_adj'] / i['num_words'], 4)
        i['adv_density'] = round(i['num_adv'] / i['num_words'], 4)


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

    def has_modifier(self):
        # nominal head may be associated with different types of modifiers and function words
        return True if self.dependency_relation in ['nmod', 'nmod:poss', 'appos', 'amod', 'nummod', 'acl', 'acl:relcl', 'det', 'clf',
                                       'case'] else False

    def __repr__(self):
        features = ['index', 'text', 'lemma', 'upos', 'xpos', 'feats', 'governor', 'dependency_relation']
        feature_str = ";".join(["{}={}".format(k, getattr(self, k)) for k in features if getattr(self, k) is not None])

        return f"<{self.__class__.__name__} {feature_str}>"
