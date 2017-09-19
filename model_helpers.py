import tensorflow as tf
import numpy as np
import os 
import sys
from ops import *
def lstm_layer(inputs, states):
    lstm = tf.nn.rnn_cell(num_units = lstm_layer_size, use_peepholes = use_peepholes)
    lstm_out = lstm(inputs, states)

def affine_layer(inputs, out_dim, name = 'affine_layer'):
    in_dim=inputs.get_shape().as_list()[1]
    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1)
        weights = tf.get_variable(name = 'weights',shape = [in_dim,out_dim]
                                , dtype = tf.float32, initializer = init)
        outputs = tf.matmul(inputs, weights)
    return outputs

def batch_norm_wrapper(inputs, is_training = True, is_ref = False, decay = 0.999, name='Bnorm'):
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
        pop_mean_summary = histogram_summary('pop_mean', pop_mean)
        pop_var_summary = histogram_summary('pop_var', pop_var)
        
            
        
            
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
        if is_ref:
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 
                                                 epsilon), pop_mean_summary, pop_var_summary
        else:
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 
                                                 epsilon)
    else:
        if is_ref:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale,
                                             epsilon), pop_mean_summary, pop_var_summary
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
            
            
            
            
