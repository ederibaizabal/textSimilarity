import stanfordnlp
from cube.api import Cube
import packageModules.Indicators
from packageModules.Transformer import Transformer


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
        self.parser = None

    '''
    Download the respective model depending of the library and language. 
    '''
    def download_model(self):
        if self.lib.lower() == "stanford":
            print("-----------You are going to use Stanford library-----------")
            if self.lang.lower() == "basque":
                print("-----------You are going to use Basque model-----------")
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

                #parser = stanfordnlp.Pipeline()

                doc = self.parser(self.text)
                # for sent in doc.sentences:
                #     for word in sent.words:
                #         print(str(
                #             word.index) + "\t" + word.text + "\t" + word.lemma + "\t" + word.upos + "\t" + word.xpos +
                #             "\t" + word.feats + "\t" + str(
                #             word.governor) + "\t" + str(word.dependency_relation) + "\n")
            else:
                print("............Working...........")
        elif self.lib.lower() == "cube":
            print("-----------You are going to use Cube Library-----------")
            if self.lang.lower() == "basque":
                cube = Cube(verbose=True)
                cube.load("eu", "latest")
                sequences = cube(self.text)
                for sequence in sequences:
                    for entry in sequence:
                        print(str(
                            entry.index) + "\t" + entry.word + "\t" + entry.lemma + "\t" + entry.upos + "\t" +
                              entry.xpos + "\t" + entry.attrs + "\t" + str(entry.head) + "\t" + str(entry.label) +
                              "\t" + entry.space_after)
            else:
                print("............Working...........")
        else:
            print("You cannot use this library. Introduce a valid library (Cube or Stanford)")

    '''
    Transform the data into a unified structure.
    '''
    def get_estructure(self):
        tf = Transformer(self.text, self.lib.lower())
        tf.get_paragraph()


prueba = NLPCharger("basque", "stanford", "Kepa hondartzan egon da.\nHurrengo astean mendira joango da. \n\nBere"
                                          " lagunak saskibaloi partidu bat antolatu dute 18etan, baina berak "
                                          "ez du jolastuko.")
prueba.download_model()
prueba.get_estructure()