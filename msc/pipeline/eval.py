import numpy as np
import pandas as pd
from scipy import stats
from itertools import groupby


def precision_at_r(df, cdist, r):
    numerator, denominator = 0, 0
    epitopes = df['epitope'].tolist()
    for i in range(len(cdist)):
        center_epitope, row = epitopes[i], cdist[i]
        for j in range(len(cdist)):
            if row[j] <= r and i != j:
                denominator += 1
                if center_epitope == epitopes[j]:
                    numerator += 1
    return {'precision': numerator/denominator,
            'avg_neighbor': denominator/len(df)}


def precision_all(df, cdist, max_radius=None):
    if max_radius is None:
        max_radius = np.max(cdist)
    numerator = np.zeros(max_radius+1, dtype=int)
    denominator = np.zeros(max_radius+1, dtype=int)
    epitopes = df['epitope'].tolist()

    for i in range(len(cdist)):
        center_epitope, row = epitopes[i], cdist[i]
        for j in range(len(cdist)):
            dist = row[j]
            if i == j or dist > max_radius:
                continue

            denominator[dist] += 1
            if center_epitope == epitopes[j]:
                numerator[dist] += 1

    numerator = np.cumsum(numerator)
    denominator = np.cumsum(denominator)
    return pd.DataFrame({'precision': numerator/denominator,
                         'avg_neighbor': denominator/len(df),
                         'radius': range(max_radius+1)})


def eval_meysman(df, cluster_col='cluster'):
    def mode(x): return stats.mode(x)[1][0]
    len_all = len(df)
    cluster = df[df[cluster_col] > -1]
    len_cluster = len(cluster)
    purity = cluster.groupby(by='cluster').agg({'epitope': mode})
    consistency = cluster.groupby(by='epitope').agg({'cluster': mode})
    return {'retention': len_cluster/len_all,
            'purity': purity.sum()[0]/len_cluster,
            'consistency': consistency.sum()[0]/len_all}
