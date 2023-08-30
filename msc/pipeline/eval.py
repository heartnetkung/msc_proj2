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
    return {'precision': numerator/denominator, 'n_prediction': denominator}


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
