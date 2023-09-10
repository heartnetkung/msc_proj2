import numpy as np
import pandas as pd
from ..neighbor_search.kdtree import kdtree


def positive(df):
    df['length'] = df['cdr3b'].str.len()
    datasets, pairs = [], []

    for length in range(11, 19):
        print(length)
        subset = df[df['length'] == length].reset_index(drop=True)
        datasets.append(subset)
        new_pair = []

        for _file in np.unique(subset['file']):
            subset2 = subset[subset['file'] == _file].reset_index(drop=True)
            new_pair += kdtree(subset2['cdr3b'])
        pairs.append(new_pair)

    return datasets, pairs
