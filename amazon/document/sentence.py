from amazon.document.interval import Interval


class Sentence(Interval):
    """ Interval corresponding to a Sentence"""

    def init(self, document, start: int, end: int):
        Interval.init(self, start, end)
        self._doc = document

    def repr(self):
        return 'Sentence({}, {})'.format(self.start, self.end)

    @property
    def tokens(self):
        """Returns the list of tokens contained in a sentence"""
        # TODO: To be implemented (tip: use Interval.overlap)