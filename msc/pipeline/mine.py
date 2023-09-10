import numpy as np
import pandas as pd
from ..neighbor_search.kdtree import kdtree
import pybktree
from pyrepseq import hamming_distance


def positive(df):
    df['length'] = df['cdr3b'].str.len()
    datasets, pairs = [], []

    for length in range(11, 18):
        print(length)
        subset = df[df['length'] == length].reset_index(drop=True)
        datasets.append(subset)
        new_pair = []

        for _file in np.unique(subset['file']):
            subset2 = subset[subset['file'] == _file].reset_index(drop=True)
            new_pair += kdtree(subset2['cdr3b'])
        pairs.append(new_pair)

    return datasets, pairs


def negative(df):
    df['length'] = df['cdr3b'].str.len()
    max_files = np.max(df['file'])
    datasets, pairs = [], []

    for length in range(11, 18):
        print(length)
        subset = df[df['length'] == length].reset_index(drop=True)
        datasets.append(subset)
        new_pair = []

        # build index
        print('x')
        lookup_index = {}
        for index, row in subset.iterrows():
            lookup_index[row['cdr3b']] = index

        # create tree
        print('y')
        useqs1 = np.unique(subset[subset['file'] < max_files/2]['cdr3b'])
        tree = pybktree.BKTree(hamming_distance, useqs1)

        # query tree
        print('z')
        for index, row in subset[subset['file'] >= max_files/2].iterrows():
            if len(new_pair)>100000:
                break
            for edit_distance, y_seq in tree.find(row['cdr3b'], 1):
                new_pair.append((index,lookup_index[y_seq]))

        print('a')
        if len(new_pair)<=100000:
            useqs1 = np.unique(subset[subset['file'] %2 ==0]['cdr3b'])
            tree = pybktree.BKTree(hamming_distance, useqs1)
            for index, row in subset[subset['file'] %2 ==1].iterrows():
                if len(new_pair)>100000:
                    break
                for edit_distance, y_seq in tree.find(row['cdr3b'], 1):
                    new_pair.append((index,lookup_index[y_seq]))

        pairs.append(new_pair)

    return datasets,pairs
