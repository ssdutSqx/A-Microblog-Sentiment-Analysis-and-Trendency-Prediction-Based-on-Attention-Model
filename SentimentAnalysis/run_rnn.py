import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
import TextRNN

time_start = time.time()
config = TextRNN.TextrnnConfig()

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
    label_array[2 * cnt + 1] = [0, 1]
    cnt += 1

train_split = np.random.randint(all, size=int(all*0.8))
train_x , train_y = vec_array[train_split], label_array[train_split]
test_x , test_y = vec_array[~train_split], label_array[~train_split]

time_end = time.time()
print("time_cost:", time_end - time_start)


def Choose_SGD(round, accuracy_list):
    if round > 500 and np.var(accuracy_list[round - 500:round]) < 5 * 1e-5:
        return True
    else:
        return False


rnn = TextRNN.TextRNN(config)
train_accuracy_list = []
test_accuracy_list = []
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(config.round):
        train_index = np.random.randint(len(train_x), size=config.mini_batch)
        if Choose_SGD(i, train_accuracy_list):
            sess.run(rnn.SGD, feed_dict={"input_x:0": train_x[train_index],"input_y:0": train_y[train_index]})
        else:
            sess.run(rnn.AdamOptimize, feed_dict={"input_x:0": train_x[train_index], "input_y:0": train_y[train_index]})

        train_accuracy = sess.run(rnn.accuracy, feed_dict={"input_x:0": train_x, "input_y:0": train_y})
        train_accuracy_list.append(train_accuracy)
        test_accuracy = sess.run(rnn.accuracy, feed_dict={"input_x:0": test_x, "input_y:0": test_y})
        test_accuracy_list.append(test_accuracy)

        if i % 10 == 0:
            print('After %d rounds,accuracy on training set is %s, on testing set is %s' % (i, train_accuracy, test_accuracy))

accuracy = pd.DataFrame(data={"train_accuracy":train_accuracy_list,"test_accuracy":test_accuracy_list})
accuracy.to_csv("LSTMModel_mini-batch=%s_hidden-size=%s_num-layers=%s_dropout-keep=%s.csv"
                % (config.mini_batch, config.hidden_size, config.num_layers, config.dropout_keep_prob), encoding="utf-8")

plt.plot(range(0, config.round), train_accuracy_list)
plt.plot(range(0, config.round), test_accuracy_list)
plt.xlabel("round")
plt.ylabel('accuracy')
plt.title('LSTM mini-batch=%s' % config.mini_batch)
plt.show()