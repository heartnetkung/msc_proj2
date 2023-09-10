import numpy as np
import pandas as pd
import time
from ..pipeline.util import read_emerson_negative,print_count, save_pd, load_pd
from ..pipeline.mine import negative

start = time.time()

#first part
# df = read_emerson_negative()
# save_pd(df, 'template_neg.csv')
# print(df.shape)
# print_count(df['cdr3b'].str.len())


df = load_pd('template_neg.csv')
sets, pairs = negative(df)
print('==============done')
for pair in pairs:
    print(len(pair))

print('elapsed time:',time.time()-start)