#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from featx import bag_of_words
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC

def read_files():
    csvfile = open('OnionOrNot.csv', 'r', encoding='UTF-8').readlines()
    x = list()
    y = list()
    for line in csvfile:
        data = str(line[:-3]).lower()
        category = line[-2]
        tokens = word_tokenize(data)
        x.append([tokens])
        y.append([category])
    z = [i == 'a' for i in y]
    return x,y,z

def split(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=109)
    return X_train, y_train

def gridsearch(train_feats, dev_feats):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear','rbf', 'poly', 'sigmoid']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(train_feats, dev_feats)
    return grid.best_estimator_

def main():
    x,y,z = read_files()
    #X_train, y_train = split(x,y)
    print(x)
    print(y)
    print(z)

    best_param = gridsearch(x, z)
    print(best_param)

main()
