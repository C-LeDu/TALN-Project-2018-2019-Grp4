from amazon.document.interval import Interval
import re


class Token(Interval):
    """ A Interval representing word like units of text with a dictionary of features """

    def __init__(self, document, start: int, end: int, shape: int, text: str, label: str):
        """
        Note that a token has 2 text representations.
        1) How the text appears in the original document e.g. doc.text[token.start:token.end]
        2) How the tokeniser represents the token e.g. nltk.word_tokenize('"') == ['``']
        :param document: the document object containing the token
        :param start: start of token in document text
        :param end: end of token in document text
        :param shape: integer label describing the shape of the token
        :param text: this is the text representation of token
        :param label: part of speach of the token
        """

        Interval.__init__(self, start, end)
        self._doc = document
        self._text = text
        self._shape = shape
        self._label = label


    @property
    def text(self):
        return self._text

    @property
    def shape(self):
        return self._shape

    @property
    def label(self):
        return self._label

    def __repr__(self):
        return 'Token({}, {}, {})'.format(self.text, self.start, self.end)

    def get_shape_category(token):
        if re.match('^[\n]+$', token):  # IS LINE BREAK
            return 'NL'
        if any(char.isdigit() for char in token) and re.match('^[0-9.,]+$', token):  # IS NUMBER (E.G., 2, 2.000)
            return 'NUMBER'
        if re.fullmatch('[^A-Za-z0-9\t\n ]+', token):  # IS SPECIAL CHARS (E.G., $, #, ., *)
            return 'SPECIAL'
        if re.fullmatch('^[A-Z\-.]+$', token):  # IS UPPERCASE (E.G., AGREEMENT, INC.)
            return 'ALL-CAPS'
        if re.fullmatch('^[A-Z][a-z\-.]+$', token):  # FIRST LETTER UPPERCASE (E.G. This, Agreement)
            return '1ST-CAP'
        if re.fullmatch('^[a-z\-.]+$', token):  # IS LOWERCASE (E.G., may, third-party)
            return 'LOWER'
        if not token.isupper() and not token.islower():  # WEIRD CASE (E.G., 3RD, E2, iPhone)
            return 'MISC'
        return 'MISC'
