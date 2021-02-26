import pandas as pd
import os
import numpy as np

res = pd.DataFrame(columns=['col_1'])
res.set_index('col_1', inplace=True)
data = {'col_1': [3, 2, 0, 0], 'col_2': ['a', 'b', 'c', 'e'], 'col_3': ['a', 'b', 'c', 'd']}
df = pd.DataFrame.from_dict(data)
df.set_index('col_1', inplace=True)
print(df)
# res = res.append(df, drop_index=True)
res = pd.concat([res, df], axis=1, join='outer')
print(res['col_2'].tolist())

data = {'col_1': [3, 1, 0], 'col_2': ['b', 'b', 'c'], 'col_4': ['a', 'b', 'e']}
df2 = pd.DataFrame.from_dict(data)
df2.set_index('col_1', inplace=True)
res = pd.concat([res, df2], axis=1, join='outer')
print(res)

def merge_fun(x):
    val = x[x.notnull()].unique()
    if len(val) > 1:
        raise Exception("Values with the same index are not matches" + val)
    return val[0] if len(val) == 1 else None
res = res.groupby(level=0, axis=1).apply(lambda x : x.apply(merge_fun, axis=1))
print(res)



