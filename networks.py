import tensorflow as tf
import numpy as np
from ops import *
import tensorflow as tf
import numpy as np
from ops import *
from model_helpers import *
from tensorflow.contrib.layers import fully_connected

class QuestionEmbeddingNet:
    def __init__(self, lstm_layer_size, num_lstm_layer, name = 'ques_embed'):
        self.name = name
        self.lstm_layer_size = lstm_layer_size
        self.num_lstm_layer = num_lstm_layer
        self.LSTMs = []
        for i in range(self.num_lstm_layer):
            self.LSTMs.append(tf.nn.rnn_cell.LSTMCell(self.lstm_layer_size, use_peepholes = True))
     
    def __call__(self, ques_inp, vocab_size, word_embed_size, max_ques_length, batch_size, 
                 is_train = True, keep_prob = 0.5,
                 scope = 'ques_embed'):
        if is_train:
            self.LSTMs_drop = []
            for i in range(self.num_lstm_layer):
                self.LSTMs_drop.append(tf.nn.rnn_cell.DropoutWrapper(self.LSTMs[i], 
                                                            output_keep_prob = keep_prob))
            self.stacked_LSTM = tf.nn.rnn_cell.MultiRNNCell(self.LSTMs_drop)
        else:
            self.stacked_LSTM = tf.nn.rnn_cell.MultiRNNCell(self.LSTMs)
        with tf.variable_scope(scope):
                init = tf.random_uniform_initializer(-0.08, 0.08)
                self.ques_embed_W = tf.get_variable(name = 'embedding_matrix', 
                                                  shape = [vocab_size, word_embed_size],
                                                  dtype = tf.float32, initializer = init)
                print('ques_embed_W', self.ques_embed_W.get_shape().as_list(),  self.ques_embed_W.dtype)
            
                states = self.stacked_LSTM.zero_state(batch_size, tf.float32)
                for i in range(1, max_ques_length + 1):
                    inputs = tf.nn.embedding_lookup(self.ques_embed_W, ques_inp[:, i-1])
                    output, states = self.stacked_LSTM(inputs, states)
                concat_list = []
                for i in range(self.num_lstm_layer):
                    concat_list.append(states[i].c)
                    concat_list.append(states[i].h)
                ques_embed = tf.concat(concat_list, axis = 1)
                print('ques_embed', ques_embed.get_shape().as_list(), ques_embed.dtype)
                return ques_embed

class ImagePlusQuesFeatureNet:
    def __init__(self, final_feat_size, activation_fn = tf.tanh):
        self.final_feat_size = final_feat_size
        self.activation_fn = activation_fn
    def __call__(self, img_inp, ques_inp):
         with tf.variable_scope('feature_combination'):
            final_img_feat = fully_connected(img_inp, self.final_feat_size, 
                                             self.activation_fn, 
                                             scope = 'img_embed_reduce_W')

            final_ques_feat = fully_connected(ques_inp, 
                                              self.final_feat_size, 
                                              self.activation_fn,
                                              scope = 'ques_embed_reduce_W')
            final_feat = tf.multiply(final_img_feat, final_ques_feat )
            
            return final_img_feat, final_ques_feat, final_feat
        
class BatchNorm:
    def __init__(self, name = 'Bnorm'):
        self.name = name
        self.pop_mean = []
        self.pop_var = []
        self.pop_mean_summ = []
        self.pop_var_summ = []
    
    def __call__(self, inputs, is_train,  decay = 0.999, name = 'Bnorm'):
        epsilon=0.001
        with tf.variable_scope(name):
            scale_init = tf.constant_initializer(1.)
            beta_init = tf.constant_initializer(0.)
            scale = tf.get_variable(name = 'scale', shape = inputs.get_shape()[-1],
                                    dtype = tf.float32, initializer = scale_init)
            beta = tf.get_variable(name = 'beta', shape = inputs.get_shape()[-1], 
                                  dtype = tf.float32, initializer = beta_init)

            pop_mean_init = tf.constant_initializer(0.)
            pop_mean = tf.get_variable(name = 'pop_mean', shape = inputs.get_shape()[1:],
                                          dtype = tf.float32, initializer = pop_mean_init, trainable = False)
            pop_var_init = tf.constant_initializer(1.)
            pop_var = tf.get_variable(name = 'pop_var', shape = inputs.get_shape()[1:],
                                       dtype = tf.float32, initializer = pop_mean_init, trainable = False)
            pop_mean_summ = histogram_summary('pop_mean', pop_mean)
            pop_var_summ = histogram_summary('pop_var', pop_var)
            
            self.pop_mean.append(pop_mean)
            self.pop_var.append(pop_var)
            self.pop_mean_summ.append(pop_mean_summ)
            self.pop_var_summ.append(pop_var_summ)




            if is_train:
                batch_mean, batch_var = tf.nn.moments(inputs,[0])
                train_mean = tf.assign(pop_mean,
                                           pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                          pop_var * decay + batch_var * (1 - decay))

                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 
                                                         epsilon)
            else:
                return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

        
        
class FeedForwardNet:
    def __init__(self,activation_fn = tf.tanh, normalization_fn = BatchNorm):
        self.batch_norm = BatchNorm()
        self.activation_fn = activation_fn
    def __call__(self, inputs, num_hidden_layer, hidden_layer_size, out_layer_size, is_train, keep_prob):
        with tf.variable_scope('FeedForwardNet'):
            output = inputs
            for i in range(num_hidden_layer):
                with tf.variable_scope('hidden' + str(i)):
                    output = affine_layer(output, hidden_layer_size)
                    output = self.batch_norm(output, is_train)
                    output = self.activation_fn(output)
                    if is_train:
                        output = tf.nn.dropout(output, keep_prob = keep_prob)
                        

            output = affine_layer(output, out_layer_size)
            return output
        