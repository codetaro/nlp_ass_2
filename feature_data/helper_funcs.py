from csv import reader
from nltk.corpus import PlaintextCorpusReader

dict_filename2category = {}
with open('labels.csv', encoding='utf-8') as f:
    for row in reader(f):
        dict_filename2category[row[1]] = row[2]

"""Helper Functions"""
def get_article_category(fileid):
    filename = fileid[:-4]
    return dict_filename2category[filename]

def get_default_dict(feature_list):
    default_dict = {}
    for feature in feature_list:
        default_dict[feature] = 0
    return default_dict

def get_feature_data(feature_list, corpus_root, file_type_regex):
    feature_data = list()
    wordlists = PlaintextCorpusReader(corpus_root, file_type_regex)
    for fileid in wordlists.fileids():
        category = get_article_category(fileid)
        feature_vector = get_default_dict(feature_list)
        for word in wordlists.words(fileid):
            if word in feature_vector.keys():
                feature_vector[word] += 1
        feature_data.append((category, feature_vector))
    return feature_data