import numpy as np
from multiprocessing import Pool
from .util import flatten_array, ensure_numpy, check_common_input, make_output
from pyrepseq import levenshtein_neighbors, hamming_neighbors

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def generate_neighbors(query, max_edits, is_hamming):
    neighbor_func = hamming_neighbors if is_hamming else levenshtein_neighbors
    ans = {query: 0}
    for edit_distance in range(1, max_edits+1):
        for seq in ans.copy():
            for new_seq in neighbor_func(seq):
                if new_seq not in ans:
                    ans[new_seq] = edit_distance
    return ans


def build_index(seqs):
    ans = {}
    for index, seq in enumerate(seqs):
        if seq not in ans:
            ans[seq] = []
        ans[seq].append(index)
    return ans


def single_lookup(args):
    index, max_edits, limit, custom_distance, max_cust_dist = _params
    ans, (x_index, seq), is_hamming = [], args, custom_distance == 'hamming'
    neighbors = generate_neighbors(seq, max_edits, is_hamming)

    for possible_edit, edit_distance in neighbors.items():
        if possible_edit in index and edit_distance <= max_edits:
            for y_index in index[possible_edit]:
                if x_index == y_index:
                    continue
                if custom_distance in (None, 'hamming'):
                    ans.append((x_index, y_index, edit_distance))
                else:
                    dist = custom_distance(seq, possible_edit)
                    if dist <= max_cust_dist:
                        ans.append((x_index, y_index, dist))
    return ans if limit is None else ans[0:limit]


def lookup(index, seqs, max_edits, max_returns, n_cpu,
           custom_distance, max_cust_dist):
    global _params
    _params = (index, max_edits, max_returns, custom_distance, max_cust_dist)

    _loop = enumerate(seqs)
    if n_cpu == 1:
        result = map(single_lookup, _loop)
    else:
        with Pool(n_cpu) as p:
            chunk = int(len(seqs)/n_cpu)
            result = p.map(single_lookup, _loop, chunksize=chunk)
    return flatten_array(result)


def brute_lookup(seqs, max_edits=1, max_returns=None, n_cpu=1,
                 custom_distance=None, max_custom_distance=float('inf'),
                 output_type='triplets'):
    """
    List all neighboring CDR3B sequences efficiently for small edit distances.
    The idea is to list all possible sequences within a given distance and lookup the dictionary if it exists.

    Parameters
    ----------
    strings : iterable of strings
        list of CDR3B sequences
    max_edits : int
        maximum edit distance defining the neighbors
    max_returns : int or None
        maximum neighbor size
    n_cpu : int
        number of CPU cores running in parallel
    custom_distance : Function(str1, str2) or "hamming"
        custom distance function to use, must statisfy 4 properties of distance (https://en.wikipedia.org/wiki/Distance#Mathematical_formalization)
    max_custom_distance : float
        maximum distance to include in the result, ignored if custom distance is not supplied
    output_type: string
        format of returns, can be "triplets", "coo_matrix", or "ndarray"

    Returns
    -------
    neighbors : array of 3D-tuples, sparse matrix, or dense matrix
        neigbors along with their edit distances according to the given output_type
        if "triplets" returns are [(x_index, y_index, edit_distance)]
        if "coo_matrix" returns are scipy's sparse matrix where C[i,j] = distance(X_i, X_j) or 0 if not neighbor
        if "ndarray" returns numpy's 2d array representing dense matrix
    """

    # boilerplate
    check_common_input(seqs, max_edits, max_returns, n_cpu,
                       custom_distance, max_custom_distance, output_type)
    seqs = ensure_numpy(seqs)

    # algorithm
    index = build_index(seqs)
    triplets = lookup(index, seqs, max_edits,
                      max_returns, n_cpu, custom_distance, max_custom_distance)
    return make_output(triplets, output_type)
