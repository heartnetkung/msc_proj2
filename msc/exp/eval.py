import numpy as np
import pandas as pd
from ..pipeline.util import read_vdj, save_np, load_np, save_pd
from ..pipeline.distance import tcrdist, levenshtein, hamming
from ..pipeline.eval import precision_at_r, precision_all
import time

df, _ = read_vdj()
start = time.time()

filename = 'eval_levenshtein.npy'
try:
    cdist1 = load_np(filename)
except FileNotFoundError as e:
    cdist1 = levenshtein(df['cdr3b'])
    save_np(cdist1, filename)
leven_df = precision_all(df, cdist1)
leven_df['distance'] = 'levenshtein'

filename = 'eval_tcrdist.npy'
try:
    cdist2 = load_np(filename)
except FileNotFoundError as e:
    cdist2 = tcrdist(df['cdr3b'])
    save_np(cdist2, filename)
tcrdist_df = precision_all(df, cdist2)
tcrdist_df['distance'] = 'tcrdist'

filename = 'eval_hamming.npy'
try:
    cdist3 = load_np(filename)
except FileNotFoundError as e:
    cdist3 = hamming(df['cdr3b'])
    save_np(cdist3, filename)
hamming_df = precision_all(df, cdist3)
hamming_df['distance'] = 'hamming'

raw_df = pd.concat([leven_df, hamming_df, tcrdist_df],
                   axis=0, ignore_index=True)
save_pd(raw_df, 'raw_eval.csv')
print('elapsed time:',time.time()-start)
