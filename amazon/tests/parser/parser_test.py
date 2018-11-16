import logging
import unittest
from amazon.parser.EnglishNerParser import EnglishNerParser
LOGGER = logging.getLogger(__name__)

filename = "testoiyer.txt"


class test_parser(unittest.TestCase):
    def setUp(self):
        with open(filename, 'r', encoding='utf-8') as fp:
            self.content = fp.read()
    # Configurer les tests unitaires (si nécessaire)

    def test_read(self):
        documents = EnglishNerParser().read(self.content)
        self.assertEqual(len(documents), 2, 'Gnééé')


if __name__ == '__main__':
    unittest.main()
