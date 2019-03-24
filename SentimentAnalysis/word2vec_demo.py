import tensorflow as tf
import jieba
import pandas as pd
import numpy as np


word_list = []
cnt=0
with open('./sgns.weibo.bigram-char','rb') as f:
    for lines in f:
        lines = lines.decode('utf-8')
        list = lines.split()
        list[1:] = [float(i) for i in list[1:]]
        if cnt == 0:
            cnt += 1
            continue
        word_list.append([list[0],list[1:]])
        # cnt += 1
        # if (cnt==1000):  break

word_list = pd.DataFrame(word_list,columns=['word','vector'],index=None)
word = word_list['word'].tolist()

seg = "我来到大学"
seg_list = jieba.lcut(seg)
print(seg_list)
sentence_vec_list = []
for item in seg_list:
    index = word.index(item) if(item in word) else -1
    if index != -1:
        vec = word_list.loc[index,'vector']
        sentence_vec_list.append(vec)
        print(vec)
sentence_vec_list = np.array(sentence_vec_list)
print(sentence_vec_list)