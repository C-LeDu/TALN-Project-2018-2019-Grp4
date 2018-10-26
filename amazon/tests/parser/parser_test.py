import logging
import unittest
import os
import random
from amazon.parser.EnglishNerParser import EnglishNerParser
LOGGER = logging.getLogger(__name__)

filename = 'F:/Utilisateur/Documents/ESGI/Cours/Traitement Automatique du Langage Naturel/data/testoiyer.txt'

class test_parser(unittest.TestCase):
    def setUp(self):
        with open(filename, 'r', encoding='utf-8') as fp:
            self.content = fp.read()
    # Configurer les tests unitaires (si nécessaire)

    def test_read(self):
        documents = EnglishNerParser().read(self.content)
        self.assertEqual(len(documents), 216, 'Gnééé')

if __name__ == '__main__':
    unittest.main()
