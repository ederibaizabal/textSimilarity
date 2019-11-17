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
        cargador = NLPCharger("basque", "stanford")
        #cargador.download_model()
        cargador.load_model()

        text = "Kepa hondartzan egon da. Eguraldi oso ona egin zuen.\nHurrengo astean mendira joango da. " \
               "\n\nBere lagunak saskibaloi partidu bat antolatu dute 18etan, baina berak ez du jolastuko. \n " \
               "Etor zaitez etxera.\n Nik egin beharko nuke lan hori. \n Gizonak liburua galdu du. \n Irten hortik!" \
                   "\n Emadazu ur botila! \n Zu beti adarra jotzen."

        document = cargador.get_estructure(text)
        indicators = document.get_indicators()
        printer = Printer(indicators)
        printer.print_info()


main = Main()
main.start()
