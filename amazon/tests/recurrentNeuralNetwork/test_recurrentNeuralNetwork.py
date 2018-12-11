import logging
import unittest

from amazon.parser.EnglishNerParser import EnglishNerParser
from amazon.vectorizer.Vectorizer import Vectorizer
from amazon.recurrentneuralnetwork.RecurrentNeuralNetwork import RecurrentNeuralNetwork

filename = "F:/Utilisateur/Documents/ESGI/Cours/Traitement Automatique du Langage Naturel/Project/TALN-Project-2018-2019-Grp4/amazon/tests/vectorizer/testoiyage.txt"


class test_recurrentNeuralNetwork(unittest.TestCase):

    def setUp(self):
        with open(filename, 'r', encoding='utf-8') as fp:
            content = fp.read()
            self.documents = EnglishNerParser().read(content)
            self.vec = Vectorizer(
                "F:/Utilisateur/Documents/ESGI/Cours/Traitement Automatique du Langage Naturel/glove.6B.50d.txt")
        # (_, self.input_shape) = self.vec.encode_features(self.documents)

    def test_build_sequence(self):
        self.assertIsInstance(RecurrentNeuralNetwork
                              .build_sequence(self.vec.word_embeddings, {"word": (10, 10), "shape": (2, 2)}, 45),
                              RecurrentNeuralNetwork)


if __name__ == '__main__':
    unittest.main()
