import tensorflow as tf


class TextrnnConfig(object):
    vector_dim = 300
    vocabulary_size = 100
    dropout_keep_prob = 0.9
    class_num = 2
    learning_rate = 1e-3
    hidden_size = 128
    num_layers = 2
    mini_batch = 256
    epoch = 10

class TextRNN:
    def __init__(self,config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None,
                                                   self.config.vocabulary_size, self.config.vector_dim], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.class_num], name="input_y")
        self.text_length = tf.placeholder(tf.int32, [None], name="text_length")

        self.rnn()

    def rnn(self):
        with tf.name_scope("bi_lstm"):
            input = self.input_x
            for _ in range(self.config.num_layers):
                with tf.variable_scope(None, default_name="bi-lstm"): # 不弄scope就会报错，dynamic_rnn 必须要在不同的scope下
                    cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)
                    cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)
                    (output, self.output_state) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, input,
                        sequence_length=self.text_length, dtype=tf.float32
                    )
                    input = tf.concat(output, 2)
            output_state = tf.concat([self.output_state[0].c, self.output_state[1].c], 1)
            print(output_state)
            self.last = tf.layers.batch_normalization(output_state, name="last")

        with tf.name_scope("output"):
            hidden_layer = tf.layers.dense(self.last, activation=tf.nn.relu, units=self.config.hidden_size,
                                                name="hidden_layer")

            '''
            with tf.variable_scope("hidden_layer", reuse=True):
                self.a_weight = tf.get_variable("kernel")
                self.a_bias = tf.get_variable("bias")
            '''

            h_drop = tf.layers.dropout(hidden_layer, self.config.dropout_keep_prob)
            score = tf.layers.dense(h_drop, units=self.config.class_num, name="scores")
            self.scores = tf.nn.softmax(score)
            self.prediction = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=tf.argmax(self.input_y, 1))
            self.loss = tf.reduce_mean(loss)
            self.AdamOptimize = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize((self.loss))
            self.SGD = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate).minimize((self.loss))

            # self.a_gradient = tf.gradients(self.loss, self.a_weight)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

