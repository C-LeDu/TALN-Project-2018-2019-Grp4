
from typing import List

from gensim.models import KeyedVectors

from amazon.document import Document
from amazon.document import Sentence

class Vectorizer:
    """ Transform a string into a vector representation"""

    def __init__(self, word_embedding_path: str):
        """
        :param word_embedding_path: path to gensim embedding file
        """
        #  Load word embeddings from file
        self.word_embeddings = KeyedVectors.load_word2vec_format(word_embedding_path, binary=False)
        #  Create shape to index dictionary

        self.shape2index = {'NL': 0, "NUMBER": 1, "SPECIAL": 2, "ALL-CAPS": 3, "1ST-CAP": 4, "LOWER": 5, "MISC": 6}
        self.shapes = self.shape2index.keys()

        # Create labels to index dictionary
        self.label2index = {'PAD': 0, 'TO': 1, 'VBN': 2, "''": 3, 'WP': 4, 'UH': 5, 'VBG': 6, 'JJ': 7, 'VBZ': 8,
                            '--': 9,
                            'VBP': 10, 'NN': 11, 'DT': 12, 'PRP': 13, ':': 14, 'WP$': 15, 'NNPS': 16, 'PRP$': 17,
                            'WDT': 18, '(': 19, ')': 20, '.': 21, ',': 22, '``': 23, '$': 24, 'RB': 25, 'RBR': 26,
                            'RBS': 27, 'VBD': 28, 'IN': 29, 'FW': 30, 'RP': 31, 'JJR': 32, 'JJS': 33, 'PDT': 34,
                            'MD': 35,
                            'VB': 36, 'WRB': 37, 'NNP': 38, 'EX': 39, 'NNS': 40, 'SYM': 41, 'CC': 42, 'CD': 43,
                            'POS': 44,
                            'LS': 45}
        self.labels = self.label2index.keys()


    def encode_features(self, documents: List[Document]):
        """
        Creates a feature matrix for all documents in the sample list
        :param documents: list of all samples as document objects
        :return: lists of numpy arrays for word, pos and shape features.
                 Each item in the list is a sentence, i.e. a list of indices (one per token)
        """
        # Loop over documents
        word = list()
        shape = list()
        for doc in documents :

            #    Loop over sentences
            for sentence in doc.sentences:
                #        Loop over tokens
                word_index_sentence = list()
                shape_index_sentence = list()
                for token in doc.tokens[sentence.start:sentence.end]:
                    #           Convert features to indices
                    if token.text.lower() in self.word_embeddings.index2word:
                        word_index_sentence.append(self.word_embeddings.index2word.index(token.text.lower()))

                    else:
                        word_index_sentence.append(0)

                    shape_index_sentence.append(self.shape2index.get(token.shape))

                #           Append to sentence
                word.append(word_index_sentence)
                shape.append(shape_index_sentence)
        return word, shape

    def encode_annotations(self, documents: List[Document]):
        """
        Creates the Y matrix representing the annotations (or true positives) of a list of documents
        :param documents: list of documents to be converted in annotations vector
        :return: numpy array. Each item in the list is a sentence, i.e. a list of labels (one per token)
        """
        labels = list()
        # Loop over documents
        for doc in documents :
            #    Loop over sentences
            for sentence in doc.sentences:
                #        Loop over tokens
                label_index_sentence = list()
                for token in doc.tokens[sentence.start:sentence.end]:
                    #           Convert label to numerical representation
                    #          And Append to labels of sentence
                    label_index_sentence.append(self.label2index.get(token.label))
                #   append to sentences
                labels.append(label_index_sentence)

        return labels



