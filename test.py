#!/usr/bin/env python3
import pandas as pd

def read_files():
    print("\n##### Reading files...")
    csvfile = open('OnionOrNot.csv', 'r', encoding='UTF-8').readlines()
    csvlist = list()
    for line in csvfile:
        line = line.strip('"')
        itemtuple = (line[-2], str(line[:-3]))
        csvlist.append(itemtuple)
    print(csvlist)
    #df = pd.read_csv('data/data.csv', index_col = 0, skiprows=1).dict()
    #for sentence,category in df:
        #print(f"{sentence} {row}")

read_files()
