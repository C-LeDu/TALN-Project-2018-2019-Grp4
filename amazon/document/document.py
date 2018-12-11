import nltk
from typing import List
from amazon.document.sentence import Sentence
from amazon.document.token import Token


class Document:
    """
    A document is a combination of text and the positions of the tags and elements in that text.
    """

    def __init__(self):
        self.text = None
        self.tokens = None
        self.sentences = None

    @classmethod
    def create_from_text(cls, text: str = None):
        """
        :param text: document text as a string
        """
        doc = Document()
        doc.text = text
        # TODO: To be implemented
        # 1. Tokenize texte (tokens & phrases)
        words, pos_tags = zip(*nltk.pos_tag(nltk.word_tokenize(text)))
        sentences = nltk.sent_tokenize(text.replace('\n', ' '))
        # 2. Trouver les intervalles de Tokens
        doc.tokens = Document._find_tokens(doc, words, pos_tags, text)
        # 3. Trouver les intervalles de phrases
        doc.sentences = Document._find_sentences(doc, sentences, text)

        return doc

    @staticmethod
    def _find_tokens(doc, word_tokens, pos_tags, text):
        """ Calculate the span of each token, find which element it belongs to and create a new Token instance
            :param doc: Reference to documents instance
            :param word_Tokens:  list of strings(tokens) coming out of nltk.word_tokenize
            :param pos_tags:  list of strings(pos tag) coming out of nltk.pos_tag
            :return: list of tokens as Token class
         """
        offset = 0
        tokens = []
        missing = None
        for token, pos_tag in zip(word_tokens, pos_tags):

            pos = text.find(token, offset, offset + max(50, len(token)))
            if pos > -1:
                shape = Token.get_shape_category(token)
                tokens.append(Token(doc, pos, pos + len(token), pos_tag, shape, token))
            else:
                raise Exception
        return tokens

    @staticmethod
    def _find_sentences(doc, sentences: List[str], doc_text: str):
        """ yield Sentence objects each time a sentence is found in the text """
        sent_list = list()
        for sent in sentences:
            pos = doc_text.find(sent)
            if pos > -1:
                s = Sentence(doc, pos, pos + len(sent))
                sent_list.append(s)
            else:
                raise Exception
        return sent_list

    @classmethod
    def create_from_vectors(cls, words: List[str], sentences: List[Sentence]=None, labels: List[str]=None):
        doc = Document()
        text = []
        offset = 0
        doc.sentences = []
        for sentence in sentences:
            text.append(' '.join(words[sentence.start:sentence.end + 1]) + ' ')
            doc.sentences.append(Sentence(doc, offset, offset + len(text[-1])))
            offset += len(text[-1])
        doc.text = ''.join(text)

        offset = 0
        doc.tokens = []
        for word, label in zip(words, labels):
            pos = doc.text.find(word, offset)
            if pos >= 0:
                offset = pos + len(word)
                doc.tokens.append(Token(doc, pos, offset, None, Token.get_shape_category(word), word, label=label))
        return doc