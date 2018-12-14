import logging
import unittest
import os
import random
from postag_eng.document import Token
LOGGER = logging.getLogger(__name__)


class test_token(unittest.TestCase):
    def setUp(self):
        self.token = Token(None, 3, 10, 7, "Bonjour", 'pos' )
    # Configurer les tests unitaires (si nécessaire)

    def test_text(self):
        self.assertEqual(self.token.text, 'Bonjour', 'Le mot \'est pas correct')

    def test_label(self):
        self.assertEqual(self.token.label, 'pos', 'Le label \'est pas correct')

    def test_shape(self):
        self.assertEqual(self.token.shape, 7, 'La shape n\'est pas correct')


if __name__ == '__main__':
    unittest.main()
