import jieba
import pandas as pd
import numpy as np
import time

time_start = time.time()

word_list = []
with open('./sgns.weibo.bigram-char','rb') as f:
    for lines in f:
        lines = lines.decode('utf-8')
        list = lines.split()
        list[1:] = [float(i) for i in list[1:]]
        word_list.append([list[0],list[1:]])

word_list = pd.DataFrame(word_list,columns=['word','vector'],index=None)
words = word_list['word'].tolist()

neg_cnt = 0
with open("negatif.txt", 'rb') as f:
    for seg in f:
        seg = seg.decode('utf-8')
        seg_list = jieba.lcut(seg)
        if len(seg_list) > 100:
            continue
        seg_vec_list = []
        for word in seg_list:
            index = words.index(word) if(word in words) else -1
            if index != -1:
                vec = word_list.loc[index,'vector']
                seg_vec_list.append(vec)
        seg_vec_array = np.array(seg_vec_list)
        name = "neg_vec_" + str(neg_cnt)
        np.save(name,seg_vec_array)
        neg_cnt += 1
print("neg_cnt:",neg_cnt)

pos_cnt = 0
with open("positif.txt", 'rb') as f:
    for seg in f:
        seg = seg.decode('utf-8')
        seg_list = jieba.lcut(seg)
        seg_vec_list = []
        if len(seg_list) > 100:
            continue
        for word in seg_list:
            index = words.index(word) if(word in words) else -1
            if index != -1:
                vec = word_list.loc[index,'vector']
                seg_vec_list.append(vec)
        seg_vec_array = np.array(seg_vec_list)
        name = "pos_vec_" + str(pos_cnt)
        np.save(name, seg_vec_array)
        pos_cnt += 1
print("pos_cnt:", pos_cnt)

time_end = time.time()
print("time_cost:",time_end - time_start)
