#!/usr/bin/python3

import nltk.classify
from nltk.tokenize import word_tokenize
from featx import bag_of_words, high_information_words, bag_of_non_stopwords, bag_of_words_in_set
from classification import precision_recall
from nltk.classify import SklearnClassifier
from sklearn.svm import LinearSVC, SVC
import string

from random import shuffle
from os import listdir  # to read files
from os.path import isfile, join  # to read files
import sys

import pandas as pd


def read_files():
    csvfile = open('OnionOrNot.csv', 'r', encoding='UTF-8').readlines()
    feats = list()
    for line in csvfile:
        line = line.strip('"')
        data = str(line[:-3])
        category = line[-2].lower()
        tokens = word_tokenize(data)
        punct = set(string.punctuation)
        for item in tokens:
            if item in punct:
                tokens.remove(item)
            else:
                pass
        bag = bag_of_non_stopwords(tokens)
        feats.append((bag, category))

    return feats


# splits a labelled dataset into two disjoint subsets train and test
def split_data(feats):
    shuffle(feats)  # randomise dataset before splitting into train and test
    split_1 = int(0.8 * len(feats))
    split_2 = int(0.9 * len(feats))
    train_feats = feats[:split_1]
    dev_feats = feats[split_1:split_2]
    test_feats = feats[split_2:]
    return train_feats, dev_feats, test_feats


# trains a classifier
def train(train_feats):
    classifier = nltk.classify.SklearnClassifier(LinearSVC(C=100))  # linear
    # classifier = nltk.classify.SklearnClassifier(SVC(kernel='poly', C=100)) #poly
    # classifier = nltk.classify.SklearnClassifier(SVC(kernel='rbf', C=100)) #rbf
    # classifier = nltk.classify.NaiveBayesClassifier.train(train_feats)
    classifier.train(train_feats)
    return classifier


def calculate_f(precisions, recalls):
    f_measures = {}
    for key in precisions.keys():
        try:
            f_measures[key] = 2 * (precisions[key] * recalls[key]) / (precisions[key] + recalls[key])
        except:
            f_measures[key] = 0
    return f_measures


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
        for w in bag.keys():
            words[category].append(w)
            all_words.append(w)


    labelled_words = [(category, words[category]) for category in categories]

    high_info_words = set(high_information_words(labelled_words, min_score=5))


    return high_info_words


# function to remove feats that are not high info
def high_info_feats(feats, high_info_words):
    high_info_feats = []
    for feat in feats:
        high_info_feats.append((bag_of_words_in_set(feat[0].keys(), high_info_words), feat[1]))
    return high_info_feats


def main():
    categories = list()
    for arg in sys.argv[1:]:
        categories.append(arg)
    feats = read_files()
    highinfo = high_information(feats, categories)
    highinfo_feats = high_info_feats(feats, highinfo)
    for item in highinfo_feats:
        print(item)
    train_feats, dev_feats, test_feats = split_data(feats)

    classifier = train(train_feats)
    score = nltk.classify.accuracy(classifier, test_feats)
    precisions, recalls = precision_recall(classifier, test_feats)
    f_measures = calculate_f(precisions, recalls)
    print(score)



if __name__ == '__main__':
    main()


