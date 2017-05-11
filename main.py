from mc_perceptron import MultiClassPerceptron

K_FOLD = 10

"""Evaluation"""
# baseline_classifier = MultiClassPerceptron.load_classifier('baseline_classifier')
# baseline_classifier.run_analytics()

# bigram_classifier = MultiClassPerceptron.load_classifier('bigram_classifier')
# bigram_classifier.run_analytics()

rake_classifier = MultiClassPerceptron.load_classifier('rake_classifier')
rake_classifier.run_analytics()

""" K-fold cross validation """
# scores = list()
# for i in range(K_FOLD):
#     classifier = MultiClassPerceptron.load_classifier('classifier_' + str(i))
#     classifier.run_analytics()
#     scores.append(classifier.calculate_accuracy())
#
# print()
# print("*** K-fold Cross Validation ***")
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.2f%%' % (sum(scores) / float(len(scores)) * 100.0))
