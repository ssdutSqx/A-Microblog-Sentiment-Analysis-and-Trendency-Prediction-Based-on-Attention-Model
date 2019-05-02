import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import Attention_LSTM

time_start = time.time()
config = Attention_LSTM.TextAT_LSTMConfig()

vec_list = []
label_list = []

neg_vec_array = np.load("neg_vec_array.npy")
for n in range(neg_vec_array.shape[0]):
    vec_list.append(neg_vec_array[n, :, :])
    label_list.append([1, 0])
del neg_vec_array

pos_vec_array = np.load("pos_vec_array.npy")
for n in range(pos_vec_array.shape[0]):
    vec_list.append(pos_vec_array[n, :, :])
    label_list.append([0, 1])
del pos_vec_array

dataset = pd.DataFrame(data = {"vector": vec_list, "label": label_list})
dataset = dataset.sample(frac=1).reset_index(drop=True)

demo = 5000
dataset = dataset.iloc[:demo]
print(dataset.head())
vec_array = np.zeros(shape=[demo, 100, 300])
label_array = np.zeros(shape=[demo, 2])
for n in range(demo):
    vec_array[n] = np.array(dataset.iloc[n, 1])
    label_array[n] = np.array(dataset.iloc[n, 0])
del dataset

train_split = np.random.randint(demo, size=int(demo*0.8))
train_x , train_y = vec_array[train_split], label_array[train_split]
test_x , test_y = vec_array[~train_split], label_array[~train_split]
del vec_array, label_array

time_end = time.time()
print("time_cost:", time_end - time_start)


def batch_index_creator(size, batch_size):
    cnt = 0
    batch_index_list = []
    while cnt < size:
        batch_index_list.append(range(cnt, min(cnt+batch_size, size)))
        cnt += batch_size
    return batch_index_list


at_lstm = Attention_LSTM.TextAT_LSTM(config)
train_accuracy_list = []
test_accuracy_list = []
train_loss_list = []
test_loss_list = []
batch_index_list = batch_index_creator(int(demo*0.8), config.mini_batch)


with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    i = 0
    for n in range(config.epoch):
        for train_index in batch_index_list:
            feed_batch = {
                "input_x:0": train_x[train_index], "input_y:0": train_y[train_index],
                "text_length:0": np.array([config.vocabulary_size] * train_x[train_index].shape[0], dtype='int32')
            }
            feed_train = {
                "input_x:0": train_x[train_index], "input_y:0": train_y[train_index],
                "text_length:0": np.array([config.vocabulary_size] * train_x[train_index].shape[0], dtype='int32')
            }
            feed_test = {
                "input_x:0": train_x[train_index], "input_y:0": train_y[train_index],
                "text_length:0": np.array([config.vocabulary_size] * train_x[train_index].shape[0], dtype='int32')
            }

            batch_loss_before = sess.run(at_lstm.loss, feed_dict=feed_batch)
            sess.run(at_lstm.AdamOptimize, feed_dict=feed_batch)
            batch_loss_after = sess.run(at_lstm.loss, feed_dict=feed_batch)

            train_loss = sess.run(at_lstm.loss, feed_dict=feed_train)
            train_loss_list.append(train_loss)
            test_loss = sess.run(at_lstm.loss, feed_dict=feed_test)
            test_loss_list.append(test_loss)

            train_accuracy = sess.run(at_lstm.accuracy, feed_dict=feed_train)
            train_accuracy_list.append(train_accuracy)
            test_accuracy = sess.run(at_lstm.accuracy, feed_dict=feed_test)
            test_accuracy_list.append(test_accuracy)

            print(i, batch_loss_before, batch_loss_after, train_loss, test_loss, train_accuracy, test_accuracy)
            i += 1

accuracy = pd.DataFrame(data={"train_accuracy":train_accuracy_list, "test_accuracy":test_accuracy_list,
                              "train_loss":train_loss_list, "test_loss":test_loss_list})
accuracy.to_csv("AT_LSTMModel_AdamOnly_mini-batch=%s_hidden-size=%s_num-layers=%s_dropout-keep=%s.csv"
                % (config.mini_batch, config.hidden_size, config.num_layers, config.dropout_keep_prob), encoding="utf-8")

plt.plot(range(0, i), train_loss_list)
plt.plot(range(0, i), test_loss_list)
plt.xlabel("round")
plt.ylabel('loss')
plt.title('AT_LSTM mini-batch=%s' % config.mini_batch)
plt.show()
