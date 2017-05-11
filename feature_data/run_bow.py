from feature_data.bow import *
from feature_data.helper_funcs import *
from mc_perceptron import *

__author__ = "Geng Yuan"

# Constants
K_FOLD = 10  # k-fold cross validation

# Extract unigram model from corpus
# unigram1000 = ChunkedCorpus(TOPDIR, chunksize=1000, ngram=1)
bigram1000 = ChunkedCorpus(TOPDIR, chunksize=1000, ngram=2)

# Read categories from labels.csv
article_classes = list(set(dict_filename2category.values()))

# Retrieve feature list from model
article_feature_list = list(bigram1000.dictionary.token2id.keys())

# Generate feature data
article_feature_data = get_feature_data(article_feature_list, "./articles_txt", ".*txt")

#-- shapes_example --#
# from feature_data.shapes_example import *
# shape_classifier = MultiClassPerceptron(shape_classes, shape_feature_list, shape_feature_data)
# shape_classifier.train()
# shape_classifier.run_analytics()
# shape_classifier.save_classifier("shape_classifier")

#-- baseline_unigram --#
# baseline_classifier = MultiClassPerceptron(article_classes, article_feature_list, article_feature_data)
# baseline_classifier.train()
# baseline_classifier.save_classifier("baseline_classifier")

#-- bigram --#
bigram_classifier = MultiClassPerceptron(article_classes, article_feature_list, article_feature_data)
bigram_classifier.train()
bigram_classifier.save_classifier("bigram_classifier")

#-- k-fold --#
# scores = list()
# for i in range(K_FOLD):
#     classifier = MultiClassPerceptron(article_classes, article_feature_list, article_feature_data)
#     classifier.train()
#     classifier.save_classifier("classifier_" + str(i))
#     classifier.run_analytics()
#     scores.append(classifier.calculate_accuracy())
