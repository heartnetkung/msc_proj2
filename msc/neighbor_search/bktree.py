import pybktree
import numpy as np
from multiprocessing import Pool
from .util import flatten_array, ensure_numpy, check_common_input, make_output
from pyrepseq import hamming_distance, levenshtein_distance
from itertools import groupby


def build_index(seqs):
    ans = {}
    for index, seq in enumerate(seqs):
        if seq not in ans:
            ans[seq] = []
        ans[seq].append(index)
    return ans


def bktree(seqs, max_edits=1):
    ans = []
    index = build_index(seqs)

    tree = pybktree.BKTree(distance, np.unique(seqs))
    for x_index, x_seq in enumerate(seqs):
        for edit_distance, y_seq in tree.find(x_seq, max_edits):
            for y_index in index[y_seq]:
                if x_index != y_index:
                    ans.append((x_index, y_index, edit_distance))
    return ans


def bktree_leven_count(seqs, max_edits, distance):
    ans = np.zeros(max_edits+1, dtype='int')
    useqs, counts = np.unique(seqs, return_counts=True)
    index = {useqs[i]: counts[i] for i in range(len(counts))}

    tree = pybktree.BKTree(distance, useqs)
    for i, x_seq in enumerate(useqs):
        for edit_distance, y_seq in tree.find(x_seq, max_edits):
            if x_seq != y_seq:
                ans[edit_distance] += counts[i] * index[y_seq]
            else:
                count = counts[i]
                ans[edit_distance] += count*(count-1)
    return ans//2


def bktree_count(seqs, max_edits=1, is_hamming=True):
    if not is_hamming:
        return bktree_leven_count(seqs, max_edits, levenshtein_distance)
    ans = np.zeros(max_edits+1, dtype='int')
    data = sorted(seqs, key=len)
    for k, g in groupby(data, key=len):
        ans += bktree_leven_count(list(g), max_edits, hamming_distance)
    return ans
