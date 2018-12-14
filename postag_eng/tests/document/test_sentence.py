
import logging
import unittest
import os
import random
from nltk.corpus import gutenberg
from postag_eng.document import Document
LOGGER = logging.getLogger(__name__)


class test_sentence(unittest.TestCase):
    def setUp(self):
        self.string = "Lorem ipsum dolor sit amet. Jesus en slip."
    # Configurer les tests unitaires (si nécessaire)

    def test_tokens(self):
        document = Document.create_from_text(self.string)
        self.assertEqual(len(document.sentences[0].tokens), 6, 'ERROR')
