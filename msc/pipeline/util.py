import time
import pandas as pd
import numpy as np
from os import path, listdir


def read_vdj(filename=None):
    if filename is None:
        filename = 'vdjdb_full.txt'
        # filename = 'vdjdb.slim.txt'
    file = path.abspath(
        path.join(__file__, '../../../data/vdjdb-2023-06-01/'+filename))
    df = pd.read_csv(file, sep='\t', keep_default_na=False)

    filter1 = df['species'] == 'HomoSapiens'
    filter2 = df['cdr3.beta'].str.match(r'^C[ACDEFGHIKLMNPQRSTVWY]{9,16}[FW]$')
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
            r'^C[ACDEFGHIKLMNPQRSTVWY]{9,16}[FW]$')]
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


def read_template():
    _file = path.abspath(path.join(__file__, '../../../data/templategt8.csv'))
    return pd.read_csv(_file, index_col=False)


def print_count(arr):
    u, c = np.unique(arr, return_counts=True)
    for i in range(len(u)):
        print(u[i], c[i])


def read_emerson_healthy(single_file=False):
    folder = path.abspath(path.join(__file__, '../../../../msc_proj/data/emerson'))
    ans = None
    files = listdir(folder)
    total_tcr = 0
    for i, file in enumerate(files):
        print(i/len(files))
        df = pd.read_csv(path.abspath(path.join(folder, file)),
                         sep='\t', usecols=['amino_acid','templates'])
        df['templates'].fillna(1,inplace=True)
        df.dropna(inplace=True)
        total_tcr += len(df)
        if len(df)==0:
            continue
        df = df[df['amino_acid'].str.match(
            r'^C[ACDEFGHIKLMNPQRSTVWY]{9,16}[FW]$')].reset_index(drop=True)

        df = df.groupby('amino_acid')['templates'].sum().reset_index()
        df['file'] = i
        filter1 = df['templates'] > 6
        filter2 = np.all((df['amino_acid'].str.len()==14,df['templates'] > 4),axis=0)
        filter3 = np.all((df['amino_acid'].str.len()>14,df['templates'] > 1),axis=0)
        df = df[np.logical_or(filter1,filter2,filter3)]
        if ans is None:
            ans = df[['amino_acid', 'file']].reset_index(drop=True)
        else:
            ans = pd.concat([ans, df], copy=False, ignore_index=True)
        if single_file:
            break
    print('total_tcr:',total_tcr)
    fieldmap = {'amino_acid': 'cdr3b', 'file': 'file'}
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
