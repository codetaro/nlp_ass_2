"""
Extract Bag-of-Words models from a corpus of text files
"""
import os
import glob
import nltk
import gensim

__author__ = "Geng Yuan"
__credits__ = "Andreas van Cranenburgh"

# A directory with .txt files
TOPDIR = './articles_txt/'

def iterdocuments(topdir):
    """Iterate over documents, yielding a list of utf8 tokens at a time."""
    for filename in sorted(glob.glob(os.path.join(topdir, '*.txt'))):
        with open(filename, encoding='utf8') as fileobj:
            document = fileobj.read()
        name = os.path.basename(filename)
        # Python 3 renamed the unicode type to str
        if isinstance(name, str):
            name = name.encode('utf8')
        tokenized = gensim.utils.tokenize(document, lowercase=True)
        yield name, tokenized

def ngrams(tokens, n):
    """Turn a sequence of tokens into space-separated n-grams"""
    if n == 1:
        return tokens
    return (' '. join(a) for a in nltk.ngrams(tokens, n))

class ChunkedCorpus(object):
    """Split text files into chunks and extract n-gram BOW model."""
    def __init__(self, topdir, chunksize=5000, ngram=1, dictionary=None):
        self.topdir = topdir
        self.ngram = ngram
        self.chunksize = chunksize
        self.chunknames = []

        if dictionary is None:
            self.dictionary = gensim.corpora.Dictionary(
                ngrams(tokens, ngram)
                for _, tokens in iterdocuments(topdir))
            self.dictionary.filter_extremes(no_below=5, keep_n=2000000)
            self.dictionary.compactify()
        else:
            self.dictionary = dictionary

    def __iter__(self):
        for filename, tokens in iterdocuments(self.topdir):
            for n, chunk in enumerate(gensim.utils.chunkize(
                    ngrams(tokens, self.ngram),
                    self.chunksize,
                    maxsize=2)):
                self.chunknames.append('%s_%d' % (filename, n))
                yield self.dictionary.doc2bow(chunk)