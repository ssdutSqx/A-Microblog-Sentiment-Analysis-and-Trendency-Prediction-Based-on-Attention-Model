import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

time_start = time.time()

class TextcnnConfig(object):
    # 参数配置
    vector_dim = 300
    vocabulary_size = 100
    filter_num = 32
    window_size = [3,4,5]
    dropout_keep_prob = 0.5
    class_num = 2
    learning_rate = 1e-3
    mini_batch = 20
    round = 1000

config = TextcnnConfig()

cnt = 0
demo = 2000
neg_full = 19226
pos_full = 18601
all = 2 * demo # neg_full + pos_full
vec_array = np.zeros(shape=[2 * demo,config.vocabulary_size,300])
label_array = np.zeros(shape=[2 * demo,2])

while cnt != demo: # neg_full = 19226

    name_neg = "neg_vec_" + str(cnt) + ".npy"
    neg_seg_vec = np.load(name_neg)
    if neg_seg_vec.shape[0] < config.vocabulary_size:
        neg_seg_vec = np.row_stack((neg_seg_vec,np.zeros([config.vocabulary_size - neg_seg_vec.shape[0],300])))
    vec_array[2 * cnt] = neg_seg_vec
    label_array[2 * cnt] = [1,0]

    cnt += 1

cnt = 0
while cnt != demo:  # pos_full = 18601
    name_pos = "pos_vec_" + str(cnt) + ".npy"
    pos_seg_vec = np.load(name_pos)
    if pos_seg_vec.shape[0] < config.vocabulary_size:
        pos_seg_vec = np.row_stack((pos_seg_vec,np.zeros([config.vocabulary_size - pos_seg_vec.shape[0],300])))
    vec_array[2 * cnt + 1] = pos_seg_vec
    label_array[2 * cnt] = [0,1]

vec_array = np.expand_dims(vec_array,-1).astype(np.float32)

train_split = np.random.randint(2 * demo,size=int(2*demo*0.8))
train_x , train_y = vec_array[train_split],label_array[train_split]
test_x , test_y = vec_array[~train_split],label_array[~train_split]

time_end = time.time()
print("time_cost:",time_end - time_start)


class TextCNN(object):
    def __init__(self,config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None,
                                                   self.config.vocabulary_size,self.config.vector_dim,1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.class_num], name="input_y")

        self.cnn()

    def cnn(self):
        pooled_outpus = []
        for filter_size in self.config.window_size:

            with tf.name_scope("conv-max_pool%s" % filter_size):

                filter_shape = [filter_size,self.config.vector_dim,1,self.config.filter_num]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape,stddev=0.1),name="W")
                b = tf.Variable(tf.constant(0.1,shape=[self.config.filter_num]),name="b")
                conv = tf.nn.conv2d(self.input_x,W,strides=[1,1,1,1],padding="VALID",name="conv")
                h = tf.tanh(tf.nn.bias_add(conv,b), name = "tanh")

                pooled = tf.nn.max_pool(h,ksize=[1,self.config.vocabulary_size - filter_size + 1
                    ,1,1],strides=[1,1,1,1],padding="VALID",name="pool")
                pooled_outpus.append(pooled)

        num_filters_total = self.config.filter_num * len(self.config.window_size)
        print(num_filters_total)
        self.h_pool = tf.concat(pooled_outpus,3)
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,self.config.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[num_filters_total,self.config.class_num],stddev=0.1),name="W")
            b = tf.Variable(tf.constant(0.1,shape=[self.config.class_num]),name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        print(self.predictions)
        print(self.scores)
        print(self.input_y)

        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=tf.argmax(self.input_y,1))
            self.loss = tf.reduce_mean(losses)
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize((self.loss))

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

cnn = TextCNN(config)
train_accuracy_list = []
test_accuracy_list = []
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(config.round):
        train_index = np.random.randint(len(train_x),size=config.mini_batch)
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

plt.plot(range(0,config.round),train_accuracy_list)
plt.plot(range(0,config.round),test_accuracy_list)
plt.xlabel("round")
plt.ylabel('accuracy')
plt.title('CNN mini-batch=%s' % config.mini_batch)
plt.show()
