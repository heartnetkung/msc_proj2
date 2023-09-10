import numpy as np
import pandas as pd
import time
from ..pipeline.util import load_np, save_np, read_emerson_healthy
from ..pipeline.util import read_vdj, save_pd, read_template, print_count
from ..pipeline.mine import positive


def preprocess():
    df = read_emerson_healthy(False)
    save_pd(df, 'templategt9.csv')


def main(df):
    # save_pd(df, 'templategt8.csv')
    # lengths = list(map(len, df['cdr3b']))
    # print_count(lengths)
    sets, pairs = positive(df)
    print('==============done')
    for pair in pairs:
        print(len(pair))


# df = read_emerson_healthy(False)
# df,_ = read_vdj() # test
# _,df = read_vdj() # val
start_time = time.time()

preprocess()
df = read_template()
print('dataframe shape', df.shape)
main(df)

print("\n--- {} seconds ---".format(time.time() - start_time))
