import time
import pandas as pd
import numpy as np
from os import path, listdir
import itertools


def read_vdj(filename=None):
    if filename is None:
        filename = 'vdjdb_full.txt'
        # filename = 'vdjdb.slim.txt'
    file = path.abspath(
        path.join(__file__, '../../../data/vdjdb-2023-06-01/'+filename))
    df = pd.read_csv(file, sep='\t', keep_default_na=False)

    filter1 = df['species'] == 'HomoSapiens'
    filter2 = df['vdjdb.score'] > 0
    filter3 = df['reference.id'] != 'PMID:28636592'
    filter4 = df['cdr3.beta'].str.match(r'^C[ACDEFGHIKLMNPQRSTVWY]{3,23}[FW]$')
    filter5 = df['v.beta'] != ''
    filter6 = df['antigen.epitope'] != ''

    all_filter = np.all(
        (filter1, filter2, filter3, filter4, filter5, filter6), axis=0)
    fieldmap = {'v.beta': 'vb', 'cdr3.beta': 'cdr3b',
                'antigen.epitope': 'epitope'}
    df2 = df[all_filter].reset_index(drop=True)
    # .drop_duplicates(ignore_index=True)
    return df2.rename(columns=fieldmap)[fieldmap.values()]


def read_emerson(single_file=False):
    folder = path.abspath(path.join(__file__, '../../../data/emerson'))
    ans = None
    for i, file in enumerate(listdir(folder)):
        df = pd.read_csv(path.abspath(path.join(folder, file)),
                         sep='\t', usecols=['amino_acid', 'v_gene'])
        df.dropna(inplace=True)
        df = df[df['amino_acid'].str.match(r'^C[ACDEFGHIKLMNPQRSTVWY]{3,23}[FW]$')]
        df = df[df['v_gene'].str.contains(r'\-\d\d$')]
        df['file'] = i
        if ans is None:
            ans = df[['amino_acid', 'file', 'v_gene']].reset_index(drop=True)
        else:
            ans = pd.concat([ans, df], copy=False, ignore_index=True)
        if single_file:
            break
    fieldmap = {'v_gene': 'vb', 'amino_acid': 'cdr3b', 'file': 'file'}
    return ans.rename(columns=fieldmap)[fieldmap.values()]


def save_np(np_arr, filename):
    file = path.abspath(
        path.join(__file__, '../../../data/'+filename))
    np.save(file, np_arr)


def load_np(filename):
    file = path.abspath(
        path.join(__file__, '../../../data/'+filename))
    return np.load(file)
