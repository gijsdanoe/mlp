#!/usr/bin/python3

# Basic classifiction functionality with naive Bayes. File provided for the assignment on classification (IR course 2019/20)

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
def split_train_test(feats, split=0.9):
    train_feats = []
    test_feats = []
    # print (feats[0])

    shuffle(feats)  # randomise dataset before splitting into train and test
    cutoff = int(len(feats) * split)
    train_feats, test_feats = feats[:cutoff], feats[cutoff:]

    print("\n##### Splitting datasets...")
    print("  Training set: %i" % len(train_feats))
    print("  Test set: %i" % len(test_feats))
    return train_feats, test_feats

# trains a classifier
def train(train_feats):
    classifier = nltk.classify.SklearnClassifier(LinearSVC(C=100))  # linear
    # classifier = nltk.classify.SklearnClassifier(SVC(kernel='poly', C=100)) #poly
    # classifier = nltk.classify.SklearnClassifier(SVC(kernel='rbf', C=100)) #rbf
    classifier.train(train_feats)
    return classifier


# classifier = nltk.classify.NaiveBayesClassifier.train(train_feats, estimator=LaplaceProbDist)


def calculate_f(precisions, recalls):
    f_measures = {}
    for key in precisions.keys():
        try:
            f_measures[key] = 2 * (precisions[key] * recalls[key]) / (precisions[key] + recalls[key])
        except:
            f_measures[key] = 0
    return f_measures


# prints accuracy, precision and recall
def evaluation(classifier, test_feats, categories):
    print("\n##### Evaluation...")
    print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))

    precisions, recalls = precision_recall(classifier, test_feats)

    f_measures = calculate_f(precisions, recalls)
    print(" |-----------|-----------|-----------|-----------|")
    print(" |%-11s|%-11s|%-11s|%-11s|" % ("category", "precision", "recall", "F-measure"))
    print(" |-----------|-----------|-----------|-----------|")
    for category in categories:
        if precisions[category] is None:
            print(" |%-11s|%-11s|%-11s|%-11s|" % (category, "NA", "NA", "NA"))

        else:

            print(" |%-11s|%-11f|%-11f|%-11s|" % (
                category, precisions[category], recalls[category], f_measures[category]))

    print(" |-----------|-----------|-----------|-----------|")



# obtain the high information words
def high_information(feats, categories):
    print("\n##### Obtaining high information words...")

    labelled_words = [(category, []) for category in categories]

    # 1. convert the formatting of our features to that required by high_information_words
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
    #		break

    labelled_words = [(category, words[category]) for category in categories]
    # print labelled_words

    # 2. calculate high information words
    high_info_words = set(high_information_words(labelled_words, min_score=5))
    # print(high_info_words)
    # high_info_words contains a list of high-information words. You may want to use only these for classification.
    # You can restrict the words in a bag of words to be in a given 2nd list (e.g. in function read_files)
    # e.g. bag_of_words_in_set(words, high_info_words)

    print("  Number of words in the data: %i" % len(all_words))
    print("  Number of distinct words in the data: %i" % len(set(all_words)))
    print("  Number of distinct 'high-information' words in the data: %i" % len(high_info_words))

    return high_info_words


# function to remove feats that are not high info
def high_info_feats(feats, high_info_words):
    high_info_feats = []
    for feat in feats:
        high_info_feats.append((bag_of_words_in_set(feat[0].keys(), high_info_words), feat[1]))
    return high_info_feats


def main():

    feats = read_files()
    print(feats)
    highinfo = high_information(feats, categories)
    highinfo_feats = high_info_feats(feats, highinfo)

    # train_feats, test_feats = split_train_test(feats)

    # TODO to use n folds you'd have to call function split_folds and have the subsequent lines inside a for loop
    nfold_feats = split_folds(highinfo_feats)
    scorelist = []
    for train_feats, test_feats in nfold_feats:
        classifier = train(train_feats)
        scorelist.append(nltk.classify.accuracy(classifier, test_feats))

    scorelist.append(sum(scorelist) / len(scorelist))
    for score in scorelist:
        print(score)


if __name__ == '__main__':
    main()


