import numpy as np
import pandas as pd
import time

time_start = time.time()
data = pd.DataFrame(columns=['vector','label'])
print(data.head())

cnt = 0
while cnt != 21576:
    name_neg = "neg_vec_" + str(cnt) + ".npy"
    name_pos = "pos_vec_" + str(cnt) + ".npy"
    neg_seg_vec = np.load(name_neg)
    pos_seg_vec = np.load(name_pos)
    data = data.append([{'vector':neg_seg_vec,'label':0}])
    data = data.append([{'vector':pos_seg_vec,'label':1}])
    cnt += 1

time_end = time.time()

data = data.reset_index(drop=True)

print(data.head())
print("time_cost:",time_end - time_start)


