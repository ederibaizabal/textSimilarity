import stanfordnlp

class Prueba:

    def __init__(self):
        print("Funciona")

    def pruebaStanford(self):
        #MODELS_DIR = '/home/edercarbajo/eu'
        MODELS_DIR = 'J:\TextSimilarity\eu'
        stanfordnlp.download('eu', MODELS_DIR)  # Download the Basque models
        # config = {'processors': 'tokenize,pos,lemma,depparse',  # Comma-separated list of processors to use
        #           'lang': 'eu',  # Language code for the language to build the Pipeline in
        #           'tokenize_model_path': '/home/edercarbajo/eu/eu_bdt_models/eu_bdt_tokenizer.pt',
        #           # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        #           'pos_model_path': '/home/edercarbajo/eu/eu_bdt_models/eu_bdt_tagger.pt',
        #           'pos_pretrain_path': '/home/edercarbajo/eu/eu_bdt_models/eu_bdt.pretrain.pt',
        #           'lemma_model_path': '/home/edercarbajo/eu/eu_bdt_models/eu_bdt_lemmatizer.pt',
        #           'depparse_model_path': '/home/edercarbajo/eu/eu_bdt_models/eu_bdt_parser.pt',
        #           'depparse_pretrain_path': '/home/edercarbajo/eu/eu_bdt_models/eu_bdt.pretrain.pt'
        #           }
        config = {'processors': 'tokenize,pos,lemma,depparse',  # Comma-separated list of processors to use
                  'lang': 'eu',  # Language code for the language to build the Pipeline in
                  'tokenize_model_path': 'J:\TextSimilarity\eu\eu_bdt_models\eu_bdt_tokenizer.pt',
                  # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
                  'pos_model_path': 'J:\TextSimilarity\eu\eu_bdt_models\eu_bdt_tagger.pt',
                  'pos_pretrain_path': 'J:\TextSimilarity\eu\eu_bdt_models\eu_bdt.pretrain.pt',
                  'lemma_model_path': 'J:\TextSimilarity\eu\eu_bdt_models\eu_bdt_lemmatizer.pt',
                  'depparse_model_path': 'J:\TextSimilarity\eu\eu_bdt_models\eu_bdt_parser.pt',
                  'depparse_pretrain_path': 'J:\TextSimilarity\eu\eu_bdt_models\eu_bdt.pretrain.pt'
                  }
        parser = stanfordnlp.Pipeline(**config)
        # parser = stanfordnlp.Pipeline()
        text = "Kepa hondartzan egon da."
        doc = parser(text)
        for sent in doc.sentences:
            for word in sent.words:
                print(str(
                    word.index) + "\t" + word.text + "\t" + word.lemma + "\t" + word.upos + "\t" + word.xpos + "\t" + word.feats + "\t" + str(
                    word.governor) + "\t" + str(word.dependency_relation) + "\n")


class Text:

    def __init__(self, text):
        self._text = text
        self._conll_file = None
        self._sentences = []

    @property
    def text(self):
        """ Access text of this document. Example: 'This is a sentence.'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the document's text value. Example: 'This is a sentence.'"""
        self._text = value

    @property
    def sentences(self):
        """ Access list of sentences for this document. """
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        """ Set the list of tokens for this document. """
        self._sentences = value


class Charger:

    def __init__(self, texto, lib):
        self = self
        self.texto = Text(
            texto)  # se considera un atributo del objeto (no se comparte con los demás objetos de esta clase)
        self.lib = lib

        # self.procesamiento()

        if lib.lower() == "stanford":
            print("You are going to use Stanford library")
        elif lib.lower() == "cube":
            print("You are going to use Cube Library")
        else:
            print("You cannot use this library. Introduce a valid library (Cube or Stanford)")

    # prueba
    def prueba(self):
        return "Hola mundo"

    # getText
    def getText(self):
        return self.texto.text

    # getSentences
    def getSentences(self):
        return self.texto.sentences

    def procesamiento(self):
        print("Processing the text...\n")


c = Charger("Yesterday we went to the beach. Today, however, we have to stay at home beacause of the stormy weather.",
            "Stanford")
# c.prueba()
print(" Txt -->  " + c.getText())
prueba = Prueba()
prueba.pruebaStanford()