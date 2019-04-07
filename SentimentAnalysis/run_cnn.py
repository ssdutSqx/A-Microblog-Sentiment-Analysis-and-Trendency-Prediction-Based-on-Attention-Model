import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
import TextCNN

time_start = time.time()

config = TextCNN.TextcnnConfig()

cnt = 0
demo = 4000
neg_full = 19226
pos_full = 18601
all = 2 * demo # neg_full + pos_full
vec_array = np.zeros(shape=[all, config.vocabulary_size,300])
label_array = np.zeros(shape=[all, 2])

while cnt != demo: # neg_full = 19226

    name_neg = "neg_vec_" + str(cnt) + ".npy"
    neg_seg_vec = np.load(name_neg)
    if neg_seg_vec.shape[0] < config.vocabulary_size:
        neg_seg_vec = np.row_stack((neg_seg_vec, np.zeros([config.vocabulary_size - neg_seg_vec.shape[0], 300])))
    vec_array[2 * cnt] = neg_seg_vec
    label_array[2 * cnt] = [1, 0]

    cnt += 1

cnt = 0
while cnt != demo:  # pos_full = 18601
    name_pos = "pos_vec_" + str(cnt) + ".npy"
    pos_seg_vec = np.load(name_pos)
    if pos_seg_vec.shape[0] < config.vocabulary_size:
        pos_seg_vec = np.row_stack((pos_seg_vec, np.zeros([config.vocabulary_size - pos_seg_vec.shape[0], 300])))
    vec_array[2 * cnt + 1] = pos_seg_vec
    label_array[2 * cnt] = [0, 1]

    cnt += 1

vec_array = np.expand_dims(vec_array, -1).astype(np.float32)

train_split = np.random.randint(all, size=int(all*0.8))
train_x , train_y = vec_array[train_split], label_array[train_split]
test_x , test_y = vec_array[~train_split], label_array[~train_split]

time_end = time.time()
print("time_cost:", time_end - time_start)



cnn = TextCNN.TextCNN(config)

train_accuracy_list = []
test_accuracy_list = []
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(config.round):
        train_index = np.random.randint(len(train_x), size=config.mini_batch)
        sess.run(cnn.optimize, feed_dict={"input_x:0": train_x[train_index], "input_y:0": train_y[train_index]})

        train_accuracy = sess.run(cnn.accuracy, feed_dict={"input_x:0": train_x, "input_y:0": train_y})
        train_accuracy_list.append(train_accuracy)
        test_accuracy = sess.run(cnn.accuracy, feed_dict={"input_x:0": test_x, "input_y:0": test_y})
        test_accuracy_list.append(test_accuracy)

        if i % 10 == 0:
            #loss = sess.run(cnn.loss,feed_dict = {"input_x:0": vec_array, "input_y:0": label_array})
            #print('After %d rounds,loss is %s' % (i,loss))
            print('After %d rounds,accuracy on training set is %s' % (i, train_accuracy))
            print('After %d rounds,accuracy on testing set is %s' % (i, test_accuracy))

plt.plot(range(0, config.round), train_accuracy_list)
plt.plot(range(0, config.round), test_accuracy_list)
plt.xlabel("round")
plt.ylabel('accuracy')
plt.title('CNN mini-batch=%s' % config.mini_batch)
plt.show()

accuracy = pd.DataFrame(data={"train_accuracy":train_accuracy_list,"test_accuracy":test_accuracy_list})
accuracy.to_csv("CnnModel_mini-batch=%s_hidden-size=%s_filter-num=%s_drop-out=%s"
                % (config.mini_batch, config.hidden_size, config.filter_num, config.dropout_keep_prob), encoding="utf-8")