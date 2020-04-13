#!/usr/bin/python3

import nltk.classify
from nltk.tokenize import word_tokenize
from featx import bag_of_words, high_information_words, bag_of_non_stopwords, bag_of_words_in_set, bag_of_bigrams_words, \
    bag_of_words_not_in_set
from nltk.classify import SklearnClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import DictVectorizer
import string


def read_files():
    csvfile = open('OnionOrNot.csv', 'r', encoding='UTF-8').readlines()
    feats = list()
    for line in csvfile:
        data = str(line[:-3]).lower()
        category = line[-2]
        if category == '0' or category == '1':
            tokens = word_tokenize(data)

            # stemming
            # stemmer = SnowballStemmer("english")

            # punctuation removal
            # tokens = [stemmer.stem(filteredtoken) for filteredtoken in tokens]
            # punct = set(string.punctuation)
            # for item in tokens:
            #     if item in punct:
            #         tokens.remove(item)
            #     else:
            #         pass

            # stopwords removal
            # tokens = bag_of_non_stopwords(tokens)

            feats.append((bag_of_words(tokens), category))
        else:
            pass
    return feats


# splits a labelled dataset into train, dev and test sets
def split_data(feats):
    split_1 = int(0.8 * len(feats))
    split_2 = int(0.9 * len(feats))
    train_feats = feats[:split_1]
    dev_feats = feats[split_1:split_2]
    test_feats = feats[split_2:]
    return train_feats, dev_feats, test_feats


# trains a classifier
def train(train_feats):
    classifier = nltk.classify.SklearnClassifier(SVC(kernel='rbf', C=100, gamma=0.001))  # SVM
    classifier.train(train_feats)
    # nltk.classify.NaiveBayesClassifier.train(train_feats)  # Naive Bayes
    return classifier


# obtain the high information words
def high_information(feats, categories):
    from collections import defaultdict
    words = defaultdict(list)
    all_words = list()
    for category in categories:
        words[category] = list()

    for feat in feats:
        category = feat[1]
        bag = feat[0]
        for w in bag:
            words[category].append(w)
            all_words.append(w)

    labelled_words = [(category, words[category]) for category in categories]
    high_info_words = set(high_information_words(labelled_words, min_score=5))
    return high_info_words


# function to remove feats that are not high info
def high_info_feats(feats, high_info_words):
    high_info_feats = []
    for feat in feats:
        highinfo = bag_of_words_in_set(feat[0], high_info_words)
        high_info_feats.append((highinfo, feat[1]))
    return high_info_feats


def bigram(feats):
    bigrams = []
    for feat in feats:
        bigramsandwords = bag_of_bigrams_words(list(feat[0].keys()))
        bigram = bag_of_words_not_in_set(bigramsandwords, list(feat[0].keys()))
        cleanbigram = list(' '.join((a, b)) for a, b in bigram)
        bigrams.append(bag_of_words(cleanbigram))
    return bigrams

def gridsearch(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear','rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def main():
    feats = read_files()

    # bigrams
    # bigrams = bigram(feats)
    # for i, j in zip(feats, bigrams):
    #     i[0].update(j)

    # high information words
    categories = ['0', '1']
    highinfo = high_information(feats, categories)
    highinfo_feats = high_info_feats(feats, highinfo)
    feats = highinfo_feats


    train_feats, dev_feats, test_feats = split_data(feats)

    classifier = train(train_feats)
    score = nltk.classify.accuracy(classifier, test_feats)
    print(score)

    # grid search
    # vectorizer = DictVectorizer()
    # X_train, y_train = list(zip(*train_feats))
    # X_train = vectorizer.fit_transform(X_train)
    # best_param = gridsearch(X_train, y_train)
    # print(best_param)



if __name__ == '__main__':
    main()