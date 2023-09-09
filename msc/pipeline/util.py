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
    filter2 = df['cdr3.beta'].str.match(r'^C[ACDEFGHIKLMNPQRSTVWY]{5,24}[FW]$')
    filter3 = df['v.beta'] != ''
    filter4 = df['antigen.epitope'] != ''
    all_filter = np.all((filter1, filter2, filter3, filter4), axis=0)
    df2 = df[all_filter].reset_index(drop=True)

    filter1 = df2['vdjdb.score'] != 0
    filter2 = df2['reference.id'] != 'PMID:28636592'
    df2_test = np.all((filter1, filter2), axis=0)
    df2_validation = ~df2_test

    fieldmap = {'v.beta': 'vb', 'cdr3.beta': 'cdr3b',
                'antigen.epitope': 'epitope'}
    df3 = df2.rename(columns=fieldmap)[fieldmap.values()]
    return df3[df2_test].reset_index(drop=True), df3[
        df2_validation].reset_index(drop=True)


def read_emerson(single_file=False):
    folder = path.abspath(path.join(__file__, '../../../data/emerson'))
    ans = None
    for i, file in enumerate(listdir(folder)):
        df = pd.read_csv(path.abspath(path.join(folder, file)),
                         sep='\t', usecols=['amino_acid', 'v_gene'])
        df.dropna(inplace=True)
        df = df[df['amino_acid'].str.match(
            r'^C[ACDEFGHIKLMNPQRSTVWY]{5,24}[FW]$')]
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


def read_emerson_healthy(single_file=False):
    folder = path.abspath(path.join(__file__, '../../../data/emerson_healthy'))
    ans = None
    for i, file in enumerate(listdir(folder)):
        df = pd.read_csv(path.abspath(path.join(folder, file)),
                         sep='\t', usecols=['amino_acid', 'v_gene'])
        df.dropna(inplace=True)
        df = df[df['amino_acid'].str.match(
            r'^C[ACDEFGHIKLMNPQRSTVWY]{5,24}[FW]$')]
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


def save_pd(df, filename):
    file = path.abspath(
        path.join(__file__, '../../../data/'+filename))
    df.to_csv(file, index=False)
