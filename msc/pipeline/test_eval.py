from .eval import eval_meysman, precision_at_r
import pandas as pd
import pyrepseq as prs


def test_eval_meysman():
    df = pd.DataFrame({
        'cluster': [0, 0, 0, 0, -1, 1, 1],
        'epitope': [0, 0, 1, 1, 1, 1, 2]
    })
    result = eval_meysman(df)
    assert result['retention'] == 6/7
    assert result['purity'] == 3/6
    assert result['consistency'] == 5/7


def test_precision():
    seqs = ['CAAA', 'CDDD', 'CADA', 'CAAK']
    cdist = prs.cdist(seqs, seqs)
    df = pd.DataFrame({'epitope': [0, 0, 0, 1]})
    assert precision_at_r(df, cdist, 1) == 6/8
