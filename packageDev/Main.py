"This is a Singleton class which is going to start necessary classes and methods."

from packageDev.Charger import NLPCharger
import re

class Main(object):

    __instance = None

    def __new__(cls):
        if Main.__instance is None:
            Main.__instance = object.__new__(cls)
        return Main.__instance

    def start(self):
        prueba = NLPCharger("basque", "stanford", "Kepa hondartzan egon da. Eguraldi oso ona egin zuen.\nHurrengo astean mendira joango da. \n\nBere"
                                          " lagunak saskibaloi partidu bat antolatu dute 18etan, baina berak "
                                          "ez du jolastuko.")
        prueba.download_model()
        struc = prueba.get_estructure()
        prueba.get_indicators(struc)


main = Main()
main.start()