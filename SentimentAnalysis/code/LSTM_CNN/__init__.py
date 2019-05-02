import tensorflow as tf


class TextLSTM_CNNConfig(object):
    vector_dim = 300
    vocabulary_size = 100
    class_num = 2
    learning_rate = 1e-4
    # basic settings

    mini_batch = 40
    epoch = 10
    # running settings

    hidden_size = 128
    num_layers = 2
    # LSTM settings

    filter_num = 80
    window_size = [3, 4, 5]
    # CNN settings

    dropout_keep_prob = 0.9
    # dense settings

class TextLSTM_CNN:
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.float32, [None,
                                                   self.config.vocabulary_size, self.config.vector_dim],name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.config.class_num], name="input_y")
        self.text_length = tf.placeholder(tf.int32, [None], name="text_length")
        self.lstm_cnn()

    def lstm_cnn(self):
        with tf.name_scope("bi_lstm"):
            input = self.input_x
            for _ in range(self.config.num_layers):
                with tf.variable_scope(None, default_name="bi-lstm"):
                    cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)
                    cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)
                    output, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, input,
                        sequence_length=self.text_length, dtype=tf.float32
                    )
                    input = output
            self.outputs = tf.expand_dims(tf.concat(input, 2), -1).astype(tf.float32)

        with tf.name_scope("cnn"):
            pooled_outputs = []
            for filter_size in self.config.window_size:
                with tf.name_scope("conv-max_pool%s" % filter_size):
                    filter_shape = [filter_size, 2*self.config.hidden_size, 1, self.config.filter_num]
                    W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), name="W")
                    b = tf.constant(0.1, shape=[self.config.filter_num], name="b")
                    conv = tf.nn.conv2d(
                        self.outputs, W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv"
                    )
                    h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")

                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, 2*self.config.hidden_size - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool"
                    )
                    pooled_outputs.append(pooled)

            num_filters_total = self.config.filter_num * len(self.config.window_size)
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

