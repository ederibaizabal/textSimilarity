import stanfordnlp
from cube.api import Cube
from packageModules.Analyzer import Analyzer
from packageModules.Transformer import Processor, ModelAdapter
from packageDev.Printer import Printer

'''
The aim of this class is the charge of the model with the specific language and nlp library.
In addition, it is going to create a unified data structure to obtain the indicators independent of the library 
and language.
'''


class NLPCharger:

    def __init__(self, language, library, text):
        self.lang = language
        self.lib = library
        self.text = text
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
                # MODELS_DIR = '/home/edercarbajo/eu'
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
                self.parser = stanfordnlp.Pipeline(**config)
            else:
                print("............Working...........")
        elif self.lib.lower() == "cube":
            print("-----------You are going to use Cube Library-----------")
            if self.lang.lower() == "basque":
                cube = Cube(verbose=True)
                cube.load("eu", "latest")
                self.parser = cube
            else:
                print("............Working...........")
        else:
            print("You cannot use this library. Introduce a valid library (Cube or Stanford)")

    '''
    Transform data into a unified structure.
    '''
    def get_estructure(self):
        tf = Processor(self.lib.lower())

        #Loading a text with paragraphs
        self.textwithparagraphs = tf.process_text(self.text)

        #Getting a unified structure [ [sentences], [sentences], ...]
        return self.adapt_nlp_model()

    def adapt_nlp_model(self):
        ma = ModelAdapter(self.parser, self.lib)
        return ma.model_analysis(self.textwithparagraphs)

    def get_indicators(self, struc):
        ind = Analyzer(self.lang, struc)
        indicators = ind.analyze()
        printer = Printer(indicators)
        printer.print_info()
