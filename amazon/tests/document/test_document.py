import logging
import unittest
import os
import random
from nltk.corpus import gutenberg

from amazon.document import Document
LOGGER = logging.getLogger(__name__)


class test_document(unittest.TestCase):
    def setUp(self):
        self.string = "Lorem ipsum dolor sit amet. Jesus en slip."
    # Configurer les tests unitaires (si nécessaire)

    def test_create_from_text(self):
        document = Document.create_from_text(self.string)
        self.assertEqual(len(document), 216, 'Some document were not extracted')

if __name__ == '__main__':
    unittest.main()
