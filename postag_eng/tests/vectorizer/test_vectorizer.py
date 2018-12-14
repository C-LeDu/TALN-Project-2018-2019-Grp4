import logging
import unittest
from postag_eng.parser.EnglishNerParser import EnglishNerParser
from postag_eng.vectorizer.Vectorizer import Vectorizer

filename = "testoiyage.txt"


class test_vectorizer(unittest.TestCase):

    def setUp(self):
        with open(filename, 'r', encoding='utf-8') as fp:
            content = fp.read()
            self.documents = EnglishNerParser().read(content)
            self.vec = Vectorizer("F:/Utilisateur/Documents/ESGI/Cours/Traitement Automatique du Langage Naturel/glove.6B.50d.txt")

    def test_vectorizer(self):

        (word, shape) = self.vec.encode_features(self.documents)
        self.assertEqual(word[0][0], 2162)
        self.assertEqual(shape[0][0], 3)

    def test_annotation(self):
        self.vec.encode_annotations(self.documents)
        label = self.vec.encode_annotations(self.documents)
        self.assertEqual(label[0][0], 38)


if __name__ == '__main__':
    unittest.main()



