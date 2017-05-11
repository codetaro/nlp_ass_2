import rake
import operator
import csv

from nltk.corpus import PlaintextCorpusReader
from feature_data.helper_funcs import *
from mc_perceptron import *

"""Read categories from labels.csv, and store into article_classes"""
with open("labels.csv", encoding='utf-8') as f:
    article_classes = list(set(row[2] for row in csv.reader(f)))

"""Feature List"""
# 1.Write the content of each document to a single file
# corpus_root = './articles_txt'
# wordlists = PlaintextCorpusReader(corpus_root, '.*.txt')
# with open('corpus.txt', 'w', encoding='utf-8') as f:
#     for fileid in wordlists.fileids():
#         f.write(wordlists.raw(fileid))

# 2.Extract keywords from corpus using RAKE
# rake_object = rake.Rake("SmartStoplist.txt", min_char_length=4, max_words_length=1, min_keyword_frequency=5)
# with open('corpus.txt', 'r', encoding='utf-8') as f:
#     corpus = f.read()
#     keywords = rake_object.run(corpus)
#
# with open('keywords_c4w1f5.csv', 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerows(keywords)

with open('keywords_c4w1f5.csv', encoding='utf-8') as f:
    article_feature_list = [row[0] for row in csv.reader(f)]

"""Feature Data - in case of boolean values, use 0 for False, 1 for True"""
article_feature_data = get_feature_data(article_feature_list, './articles_txt', '.*.txt')

"""Rake"""
rake_classifier = MultiClassPerceptron(article_classes, article_feature_list, article_feature_data)
rake_classifier.train()
rake_classifier.save_classifier('rake_classifier')