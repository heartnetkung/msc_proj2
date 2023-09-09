import pwseqdist as pw
from pwseqdist.metrics import nb_vector_tcrdist, nb_vector_editdistance
import numpy as np
import pandas as pd
from Levenshtein import hamming as hm


def tcrdist(seqs, seqs2=None, ncpus=1):
    args = {'seqs2': seqs2, 'use_numba': True,
            'uniqify': False, 'ncpus': ncpus}
    return pw.apply_pairwise_rect(nb_vector_tcrdist, seqs, **args)


def levenshtein(seqs, seqs2=None, ncpus=1):
    args = {'seqs2': seqs2, 'use_numba': True,
            'uniqify': False, 'ncpus': ncpus}
    return pw.apply_pairwise_rect(nb_vector_editdistance, seqs, **args)


def hamming(seqs):
    _len = len(seqs)
    ans = np.full((_len, _len), 30)
    for i in range(_len):
        for j in range(_len):
            x, y = seqs[i], seqs[j]
            if i == j:
                ans[i][j] = 0
                break
            if len(x) != len(y):
                continue
            if i == j:
                pass
            ans[i][j] = hm(x, y)
    return ans
