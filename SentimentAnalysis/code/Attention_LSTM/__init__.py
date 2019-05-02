import tensorflow as tf


class TextAT_LSTMConfig(object):
    vector_dim = 300
    vocabulary_size = 100
    dropout_keep_prob = 0.9
    class_num = 2
    learning_rate = 5 * 1e-5
    hidden_size = 128
    num_layers = 1
    mini_batch = 40
    epoch = 10

class TextAT_LSTM:
    def __init__(self,config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None,
                                                   self.config.vocabulary_size, self.config.vector_dim], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.class_num], name="input_y")
        self.text_length = tf.placeholder(tf.int32, shape=(None), name="text_length")

        self.at_lstm()

    def at_lstm(self):
        with tf.name_scope("bi_lstm"):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.input_x,
                sequence_length=self.text_length, dtype=tf.float32
            )
            outputs = tf.concat(outputs, 2)
            self.outputs = tf.layers.dense(outputs, units=self.config.hidden_size)

        with tf.name_scope("attention"):
            M = tf.nn.tanh(tf.layers.batch_normalization(self.outputs), name="M")
            alpha = tf.nn.softmax(tf.layers.dense(M, units=1), name="alpha")
            r = tf.matmul(tf.transpose(self.outputs, [0, 2, 1]), alpha)
            self.r = tf.layers.batch_normalization(tf.reshape(r, [-1, self.config.hidden_size]), name="r")


        with tf.name_scope("output"):
            hidden_layer = tf.layers.dense(self.r, activation=tf.nn.relu, units=self.config.hidden_size,
                                                name="hidden_layer")
            h_drop = tf.layers.dropout(hidden_layer, self.config.dropout_keep_prob)

            score = tf.layers.dense(h_drop, units=self.config.class_num, name="scores")
            self.scores = tf.nn.softmax(score)
            self.prediction = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=tf.argmax(self.input_y, 1))
            self.loss = tf.reduce_mean(loss)
            self.AdamOptimize = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize((self.loss))
            self.SGD = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate).minimize((self.loss))

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")