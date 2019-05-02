import tensorflow as tf

class TextcnnConfig(object):
    # 参数配置
    vector_dim = 300
    vocabulary_size = 100
    filter_num = 80
    window_size = [3, 4, 5]
    dropout_keep_prob = 0.8
    class_num = 2
    learning_rate = 1e-3
    hidden_size = 128
    mini_batch = 50
    epoch = 10

class TextCNN(object):
    def __init__(self,config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None,
                                                   self.config.vocabulary_size, self.config.vector_dim,1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.class_num], name="input_y")

        self.cnn()

    def cnn(self):
        pooled_outputs = []
        for filter_size in self.config.window_size:
            with tf.name_scope("conv-max_pool%s" % filter_size):
                filter_shape = [filter_size, self.config.vector_dim, 1, self.config.filter_num]
                W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.filter_num]), name="b")
                conv = tf.nn.conv2d(
                    self.input_x, W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                h = tf.tanh(tf.nn.bias_add(conv, b), name="tanh")

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.vocabulary_size-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool"
                )
                pooled_outputs.append(pooled)

        num_filters_total = self.config.filter_num * len(self.config.window_size)
        print(num_filters_total)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("output"):
            # 全连接层
            self.hidden_layer = tf.layers.dense(self.h_pool_flat, activation=tf.nn.relu
                                                , units=self.config.hidden_size, name="hidden_layer")
                    # dense的作用 添加一个隐含层
            self.h_drop = tf.nn.dropout(self.hidden_layer, self.config.dropout_keep_prob)

            # 输出
            score = tf.layers.dense(self.h_drop, units=self.config.class_num, name="scores")
            self.scores = tf.nn.softmax(score)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        print(self.scores)

        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=tf.argmax(self.input_y, 1))
            self.loss = tf.reduce_mean(losses)
            self.AdamOptimize = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize((self.loss))
            self.SGD = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate).minimize((self.loss))

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")