from postag_eng.document import Sentence
from postag_eng.document.interval import Interval
from postag_eng.document.document import Document
from postag_eng.parser.parser import Parser


class EnglishNerParser(Parser):
    def read(self, content: str):
        """Reads the content of a NER/POS data file and returns one document instance per document it finds."""
        documents = []
        # 1. Split the text in documents using string '-DOCSTART- -X- O O' and loop over it
        content = content.split('-DOCSTART- -X- O O')
        for doc in content:
            if doc != '':
                words = []
                sentences = []
                labels = []
                start = 0
                # 2. Split lines and loop over
                str_sentences = doc.split('\n\n')
                # 3. Make vectors of tokens and labels (colunn 4) and at the '\n\n' make a sentence
                for sentence in str_sentences:
                    if sentence != '':
                        tokens = sentence.split('\n')
                        for token in tokens:
                            cols = token.split(' ')
                            words.append(cols[0])
                            labels.append(cols[1])
                        sentences.append(Sentence(doc, start, start+len(tokens)))
                        start += len(tokens)
                # 4. Create a Document object
                documents.append(Document.create_from_vectors(words, sentences, labels))

        return documents
