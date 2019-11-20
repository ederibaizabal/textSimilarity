import stanfordnlp
from cube.api import Cube
from packageModules.Transformer import ModelAdapter
from packageDev.Printer import Printer
import re

'''
The aim of this class is the charge of the model with the specific language and nlp library.
In addition, it is going to create a unified data structure to obtain the indicators independent of the library 
and language.
'''


class NLPCharger:

    def __init__(self, language, library):
        self.lang = language
        self.lib = library
        self.text = None
        self.textwithparagraphs = None
        self.parser = None

    '''
    Download the respective model depending of the library and language. 
    '''
    def download_model(self):
        if self.lib.lower() == "stanford":
            print("-----------You are going to use Stanford library-----------")
            if self.lang.lower() == "basque":
                print("-------------You are going to use Basque model-------------")
                # MODELS_DIR = '/home/kepa/eu'
                MODELS_DIR = 'J:\TextSimilarity\eu'
                stanfordnlp.download('eu', MODELS_DIR)  # Download the Basque models
            elif self.lang.lower() == "english":
                print("-------------You are going to use English model-------------")
                MODELS_DIR = '/home/kepa/en'
                print("-------------Downloading Stanford Basque model-------------")
                stanfordnlp.download('en', MODELS_DIR)  # Download the Basque models
            elif self.lang.lower() == "spanish":
                print("-------------You are going to use Spanish model-------------")
                MODELS_DIR = '/home/kepa/es'
                stanfordnlp.download('es', MODELS_DIR) # Download the English models
            else:
                print("........You cannot use this language...........")
        elif self.lib.lower() == "cube":
            print("-----------You are going to use Cube Library-----------")
            if self.lang.lower() == "basque":
                cube = Cube(verbose=True)
                cube.load("eu","latest")
            elif self.lang.lower() == "english":
                cube = Cube(verbose=True)
                cube.load("en","latest")
            elif self.lang.lower() == "spanish":
                cube=Cube(verbose=True)
                cube.load("es","latest")
            else:
                print("........You cannot use this language...........")
        else:
            print("You cannot use this library. Introduce a valid library (Cube or Stanford)")
    '''
    load model in parser object 
    '''
    def load_model(self):
        if self.lib.lower() == "stanford":
            print("-----------You are going to use Stanford library-----------")
            if self.lang.lower() == "basque":
                print("-------------You are going to use Basque model-------------")
                # config = {'processors': 'tokenize,pos,lemma,depparse',  # Comma-separated list of processors to use
                #            'lang': 'eu',  # Language code for the language to build the Pipeline in
                #            'tokenize_model_path': '/home/kepa/eu/eu_bdt_models/eu_bdt_tokenizer.pt',
                #            # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
                #            'pos_model_path': '/home/kepa/eu/eu_bdt_models/eu_bdt_tagger.pt',
                #            'pos_pretrain_path': '/home/kepa/eu/eu_bdt_models/eu_bdt.pretrain.pt',
                #            'lemma_model_path': '/home/kepa/eu/eu_bdt_models/eu_bdt_lemmatizer.pt',
                #            'depparse_model_path': '/home/kepa/eu/eu_bdt_models/eu_bdt_parser.pt',
                #            'depparse_pretrain_path': '/home/kepa/eu/eu_bdt_models/eu_bdt.pretrain.pt'
                #            }
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
                self.parser = stanfordnlp.Pipeline(**config)

            elif self.lang.lower() == "english":
                print("-------------You are going to use English model-------------")
                config = {'processors': 'tokenize,mwt,pos,lemma,depparse', # Comma-separated list of processors to use
                            'lang': 'en', # Language code for the language to build the Pipeline in
                            'tokenize_model_path': '/home/kepa/en/en_ewt_models/en_ewt_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
                            #'mwt_model_path': './fr_gsd_models/fr_gsd_mwt_expander.pt',
                            'pos_model_path': '/home/kepa/en/en_ewt_models/en_ewt_tagger.pt',
                            'pos_pretrain_path': '/home/kepa/en/en_ewt_models/en_ewt.pretrain.pt',
                            'lemma_model_path': '/home/kepa/en/en_ewt_models/en_ewt_lemmatizer.pt',
                            'depparse_model_path': '/home/kepa/en/en_ewt_models/en_ewt_parser.pt',
                            'depparse_pretrain_path': '/home/kepa/en/en_ewt_models/en_ewt.pretrain.pt'
                            }
                self.parser = stanfordnlp.Pipeline(**config)
            elif self.lang.lower() == "spanish":
                print("-------------You are going to use Spanish model-------------")
                config = {'processors': 'tokenize,pos,lemma,depparse', # Comma-separated list of processors to use
                            'lang': 'es', # Language code for the language to build the Pipeline in
                            'tokenize_model_path': '/home/kepa/es/es_ancora_models/es_ancora_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
                            'pos_model_path': '/home/kepa/es/es_ancora_models/es_ancora_tagger.pt',
                            'pos_pretrain_path': '/home/kepa/es/es_ancora_models/es_ancora.pretrain.pt',
                            'lemma_model_path': '/home/kepa/es/es_ancora_models/es_ancora_lemmatizer.pt',
                            'depparse_model_path': '/home/kepa/es/es_ancora_models/es_ancora_parser.pt',
                            'depparse_pretrain_path': '/home/kepa/es/es_ancora_models/es_ancora.pretrain.pt'
                            }
                self.parser = stanfordnlp.Pipeline(**config)
            else:
                print("........You cannot use this language...........")
        elif self.lib.lower() == "cube":
            print("-----------You are going to use Cube Library-----------")
            if self.lang.lower() == "basque":
                #load(self, language_code, version="latest",local_models_repository=None,
                #local_embeddings_file=None, tokenization=True, compound_word_expanding=False,
                #tagging=True, lemmatization=True, parsing=True).
                #Ejemplo:load("es",tokenization=False, parsing=False)
                cube = Cube(verbose=True)
                cube.load("eu","latest")
            elif self.lang.lower() == "english":
                cube = Cube(verbose=True)
                cube.load("en","latest")
            elif self.lang.lower() == "spanish":
                cube=Cube(verbose=True)
                cube.load("es","latest")
            else:
                print("........You cannot use this language...........")
        else:
            print("You cannot use this library. Introduce a valid library (Cube or Stanford)")

    def process_text(self, text):
        self.text = text.replace('\n', '@')
        self.text = re.sub(r'@+', '@', self.text)
        return self.text

    '''
    Transform data into a unified structure.
    '''
    def get_estructure(self, text):
        self.text = text
        #Loading a text with paragraphs
        self.textwithparagraphs = self.process_text(self.text)

        #Getting a unified structure [ [sentences], [sentences], ...]
        return self.adapt_nlp_model()

    def adapt_nlp_model(self):
        ma = ModelAdapter(self.parser, self.lib)
        return ma.model_analysis(self.textwithparagraphs)



