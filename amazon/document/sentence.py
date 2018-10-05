from amazon.document.interval import Interval

class Sentence(Interval):
    """ Interval corresponding to a Sentence"""

    def __init__(self, document, start: int, end: int):
        Interval.__init__(self, start, end)
        self._doc = document

    def repr(self):
        return 'Sentence({}, {})'.format(self.start, self.end)

    @property
    def tokens(self):
        liste = []
        """Returns the list of tokens contained in a sentence"""
        for token in self._doc.tokens:
            if self.overlaps(token):
                liste.append(token)
        return liste
