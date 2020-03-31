#!/usr/bin/env python3
import pandas as pd

def read_files():
    print("\n##### Reading files...")
    csvfile = open('data/data.csv', 'r', encoding='UTF-8').readlines()
    csvdict = tuple()
    for line in csvfile:
        print(line[-2])
    #df = pd.read_csv('data/data.csv', index_col = 0, skiprows=1).dict()
    #for sentence,category in df:
        #print(f"{sentence} {row}")

read_files()
