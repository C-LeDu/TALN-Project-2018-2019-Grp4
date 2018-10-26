import logging
import unittest
import os
import random
from amazon.document import Token
LOGGER = logging.getLogger(__name__)


class test_parser(unittest.TestCase):
    def setUp(self):
        self.token = Token(None, 3, 10, 'pos', 7, "Bonjour" )
    # Configurer les tests unitaires (si nécessaire)

    def test_text(self):
        self.assertEqual(self.token.text,'Bonjour', 'Le mot \'est pas correct')

    def test_pos(self):
        self.assertEqual(self.token.pos,'pos', 'La pos n\'est pas correct')

    def test_shape(self):
        self.assertEqual(self.token.shape, 7 , 'La shape n\'est pas correct')

if __name__ == '__main__':
    unittest.main()
