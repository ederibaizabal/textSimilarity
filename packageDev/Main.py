"This is a Singleton class which is going to start necessary classes and methods."

from packageDev.Charger import NLPCharger
from packageDev.Printer import Printer


class Main(object):
    # _instance es un atributo que conoce el Singleton
    __instance = None

    def __new__(cls):
        if Main.__instance is None:
            Main.__instance = object.__new__(cls)
        return Main.__instance

    def start(self):
        language = "basque"
        model = "stanford"
        cargador = NLPCharger(language, model)
        cargador.download_model()
        cargador.load_model()
        if language == "basque":
            text = "Kepa hondartzan egon da. Eguraldi oso ona egin zuen.\nHurrengo astean mendira joango da. " \
                   "\n\nBere lagunak saskibaloi partidu bat antolatu dute 18etan, baina berak ez du jolastuko. \n " \
                   "Etor zaitez etxera.\n Nik egin beharko nuke lan hori. \n Gizonak liburua galdu du. \n Irten hortik!" \
                   "\n Emadazu ur botila! \n Zu beti adarra jotzen."
        if language == "english":
            text = "Kepa is going to the beach. I am Kepa. \n" \
                   "Eder is going too. He is Eder."
        if language == "spanish":
            text = "Kepa va ir a la playa. Yo soy Kepa. \n" \
                   "Ibon tambien va a ir. El es Ibon."

        document = cargador.get_estructure(text)
        indicators = document.get_indicators()
        printer = Printer(indicators)
        printer.print_info()


main = Main()
main.start()
