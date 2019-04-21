import tensorflow as tf

class TextrnnConfig(object):
    vector_dim = 300
    vocabulary_size = 100
    dropout_keep_prob = 0.8
    class_num = 2
    learning_rate = 1e-3
    hidden_size = 128
    num_layers = 2
    mini_batch = 100
    round = 5000

class TextRNN:
    def __init__(self,config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None,
                                                   self.config.vocabulary_size, self.config.vector_dim], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.class_num], name="input_y")

        self.rnn()

    def rnn(self):
        def get_a_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)


        with tf.name_scope("rnn"):
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell()for _ in range(self.config.num_layers)])
            outputs, state = tf.nn.dynamic_rnn(cell, self.input_x, dtype=tf.float32)
            print(outputs)
            last = outputs[:, -1, :]
            print(last)

        with tf.name_scope("output"):
            self.hidden_layer = tf.layers.dense(last, activation=tf.nn.relu, units=self.config.hidden_size,
                                                name="hidden_layer")
            self.h_drop = tf.layers.dropout(self.hidden_layer, self.config.dropout_keep_prob)

            score = tf.layers.dense(self.h_drop, units=self.config.class_num, name="scores")
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
