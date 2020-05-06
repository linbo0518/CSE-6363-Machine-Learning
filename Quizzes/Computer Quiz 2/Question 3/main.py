import numpy as np
import pandas as pd
from utils import distance

MAVID = 1001778270

df = pd.read_csv("NBA_0.txt", sep=' ')
df_ = pd.read_csv("NBA_0.txt", sep=' ')
df.pop("Player")
mp_value = df.pop("MP")
df = df.to_numpy()
instances_with_miss_value = df[:10]
instances = df[10:]

for miss_idx in range(10):
    dist = []
    fill_value = 0
    for idx in range(len(instances)):
        dist.append(
            distance(instances_with_miss_value[miss_idx], instances[idx]))
    top_3 = np.argsort(dist)[:3]
    for top_idx in top_3:
        fill_value += float(mp_value[top_idx + 10])
    print(
        f"{miss_idx+1:02}: {fill_value/3:.5f}, filled by indexes {top_3+10} and their distance are {np.array(dist)[top_3]}"
    )
    df_['MP'][miss_idx] = round(fill_value / 3, 5)

df_.to_csv("filled.txt",
           sep=' ',
           float_format="%.3f",
           header=False,
           index=False)
