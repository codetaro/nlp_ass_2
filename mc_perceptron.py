"""
COMP5046 Natural Language Processing
Assignment Two - Perceptron Article Classification
"""
import numpy as np
from sklearn.externals import joblib
import random

__author__ = "Geng Yuan"
__credits__ = "Sidd Karamcheti"

# Constants
BIAS = 1
TRAIN_TEST_RATIO = 0.75
ITERATIONS = 100
OUTPUT_PATH = "../classifier_models/"  # called in ./feature_data/, go one level back to find ./classifier_models/


class MultiClassPerceptron():
    # Analytics values
    precision, recall, accuracy, fbeta_score = {}, {}, 0, {}

    """
    :param classes              List of categories/classes (match tags in tagged data)
    :param feature_list         List of features
    """

    def __init__(self, classes, feature_list, feature_data, train_test_ratio=TRAIN_TEST_RATIO, iterations=ITERATIONS):
        self.classes = classes
        self.feature_list = feature_list
        self.feature_data = feature_data
        self.ratio = train_test_ratio
        self.iterations = iterations

        # Split feature data into train set, and test set
        random.shuffle(self.feature_data)
        self.train_set = self.feature_data[:int(len(self.feature_data) * self.ratio)]
        self.test_set = self.feature_data[int(len(self.feature_data) * self.ratio):]

        # Initialize empty weight vectors, with extra BIAS term
        self.weight_vectors = {c: np.array([0 for _ in range(len(feature_list) + 1)]) for c in self.classes}

    def train(self):
        for _ in range(self.iterations):
            for category, feature_dict in self.train_set:
                # Format feature values as a vector, with extra BIAS term
                feature_list = [feature_dict[k] for k in self.feature_list]
                feature_list.append(BIAS)
                feature_vector = np.array(feature_list)

                # Initialize arg_max value, predicted class
                arg_max, predicted_class = 0, self.classes[0]

                # Multi-class decision rule:
                for c in self.classes:
                    current_activation = np.dot(feature_vector, self.weight_vectors[c])
                    if current_activation >= arg_max:
                        arg_max, predicted_class = current_activation, c

                # Update rule:
                if not (category == predicted_class):
                    self.weight_vectors[category] += feature_vector
                    self.weight_vectors[predicted_class] -= feature_vector

    def predict(self, feature_dict):
        """
        Categorize an unseen data point based on the existing collected data

        :return: Return the predicted category for the data point
        """
        feature_list = [feature_dict[k] for k in self.feature_list]
        feature_list.append(BIAS)
        feature_vector = np.array(feature_list)

        # Initialize arg_max, predicted class
        arg_max, predicted_class = 0, self.classes[0]

        # Multi-class decision rule:
        for c in self.classes:
            current_activation = np.dot(feature_vector, self.weight_vectors[c])
            if current_activation >= arg_max:
                arg_max, predicted_class = current_activation, c

        return predicted_class

    def run_analytics(self):
        """
        Runs analytics on the classifier, returning data on precision, recall, accuracy, as well
        as the fbeta score

        :return: Prints statistics to screen
        """
        print()
        print("CLASSIFIER ANALYSIS: ")
        print()
        self.calculate_precision()
        print()
        self.calculate_recall()
        print()
        self.calculate_fbeta_score()
        print()
        print("=== Accuracy ===")
        print("Model Accuracy:", self.calculate_accuracy())

    def calculate_precision(self):
        """
        Calculates the precision of the classifier by running algorithm against test set and comparing
        the output to the actual categorization
        """
        test_classes = [f[0] for f in self.test_set]
        correct_counts = {c: 0 for c in test_classes}
        total_counts = {c: 0 for c in test_classes}

        for feature_dict in self.test_set:
            actual_class = feature_dict[0]
            predicted_class = self.predict(feature_dict[1])

            if actual_class == predicted_class:
                correct_counts[actual_class] += 1
                total_counts[actual_class] += 1
            else:
                total_counts[predicted_class] += 1

        print("=== Precision Statistics ===")
        for c in correct_counts:
            try:
                if not total_counts[c] == 0:
                    self.precision[c] = (correct_counts[c] * 1.0) / (total_counts[c] * 1.0)
                    print("%s class precision:" % (c.upper()), self.precision[c])
                else:
                    print("%s class precision:" % (c.upper()), "N/A")
            except KeyError:
                continue    # predicted class may be not int test_classes

    def calculate_recall(self):
        """
        Calculates the recall of the classifier by running algorithm against test set and comparing
        the output to the actual categorization
        """
        test_classes = [f[0] for f in self.test_set]
        correct_counts = {c: 0 for c in test_classes}
        total_counts = {c: 0 for c in test_classes}

        for feature_dict in self.test_set:
            actual_class = feature_dict[0]
            predicted_class = self.predict(feature_dict[1])

            if actual_class == predicted_class:
                correct_counts[actual_class] += 1
                total_counts[actual_class] += 1
            else:
                total_counts[actual_class] += 1

        print("=== Recall Statistics ===")
        for c in correct_counts:
            if not total_counts[c] == 0:
                self.recall[c] = (correct_counts[c] * 1.0) / (total_counts[c] * 1.0)
                print("%s class recall:" % (c.upper()), self.recall[c])
            else:
                print("%s class recall:" % (c.upper()), "N/A")

    def calculate_fbeta_score(self):
        """
        Calculated by taking the harmonic mean of the precision and recall values
        """
        print("=== F-beta Scores ===")
        for c in self.precision:
            try:
                self.fbeta_score[c] = 2 * ((self.precision[c] * self.recall[c]) / (self.precision[c] + self.recall[c]))
                print("%s class F-Beta score:" % (c.upper()), self.fbeta_score[c])
            except ZeroDivisionError:
                print("%s class F-Beta score:" % (c.upper()), "N/A")

    def calculate_accuracy(self):
        correct, incorrect = 0, 0
        for feature_dict in self.test_set:
            actual_class = feature_dict[0]
            predicted_class = self.predict(feature_dict[1])

            if actual_class == predicted_class:
                correct += 1
            else:
                incorrect += 1
        accuracy = (correct * 1.0) / ((correct + incorrect) * 1.0)
        return accuracy

    def save_classifier(self, classifier_name):
        """
        Saves classifier as a .pickle file to the classifier_models directory

        :param classifier_name: Name under which to save the classifier
        """
        filename = OUTPUT_PATH + classifier_name + ".sav"
        joblib.dump(self, filename)

    @staticmethod
    def load_classifier(classifier_name):
        """
        Unpickle the classifier, returns the MultiClassPerceptron object

        :param classifier_name: 
        :return: Return instance of MultiClassPerceptron
        """
        filename = OUTPUT_PATH[1:] + classifier_name + ".sav"   # called in main.py, fix the OUTPUT_PATH accordingly
        return joblib.load(filename)
