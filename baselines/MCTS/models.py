import numpy as np
import tensorflow as tf
import math
import tensorflow.contrib.layers as layers
from utils import conv, fc, conv_to_fc

def model(X, nact, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=8, stride=1, init_scale=np.sqrt(2)) # TODO: when upgraded to batch run, add layer_norm to conv
        # x = layers.layer_norm(x, scale=True, center=True)
        h2 = conv(h, 'c2', nf=64, rf=4, stride=1, init_scale=np.sqrt(2)) 
        h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)) 
        h3 = conv_to_fc(h3)
        h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
        pi = fc(h4, 'pi', nact, act=lambda x: x)
        vf = fc(h4, 'v', 1, act=lambda x: tf.tanh(x))

        # TODO: if change SAT to boolean type, modify the filter as well
        pos = tf.reduce_max(X, axis = 1) # Comments by Fei: get 1 if the postive variable exists in any clauses, otherwise 0
        neg = tf.reduce_min(X, axis = 1) # Comments by Fei: get -1 if the negative variables exists in any clauses, otherwise 0
        ind = tf.concat([pos, neg], axis = 2) # Comments by Fei: get (1, -1) if this var is present, (1, 0) if only as positive, (0, -1) if only as negative
        ind_flat = tf.reshape(ind, [-1, nact]) # Comments by Fei: this is nbatch * nact, with 0 values labeling non_valid actions, 1 or -1 for other
        ind_flat_filter = tf.abs(tf.cast(ind_flat, tf.float32)) # Comments by Fei: this is nbatch * nact, with 0 values labeling non_valid actions, 1 for other
        #pi_fil = pi + (ind_flat_filter - tf.ones(tf.shape(ind_flat_filter))) * 1e32
        pi_fil = pi + (ind_flat_filter - tf.ones(tf.shape(ind_flat_filter))) * 1e32
    return pi_fil, vf[:, 0]

def model2(X, nact, scope, reuse = False, layer_norm = False):
    # X should be nbatch * ncol * nrow * 2 (boolean)
    with tf.variable_scope(scope, reuse = reuse):
        h = conv(tf.cast(X, tf.float32), 'c1', nf = 32, rf = 8, stride = 1, init_scale = np.sqrt(2))
        # x = layers.layer_norm(x, scale = True, center = True)
        h2 = conv(h, 'c2', nf = 64, rf = 4, stride = 1, init_scale = np.sqrt(2))
        h3 = conv(h2, 'c3', nf = 64, rf = 3, stride = 1, init_scale = np.sqrt(2))
        h3 = conv_to_fc(h3)
        h4 = fc(h3, 'fc1', nh = 512, init_scale = np.sqrt(2))
        pi = fc(h4, 'pi', nact, act = lambda x : x)
        vf = fc(h4, 'v', 1, act = lambda x : tf.tanh(x))

        # filter out non-valid actions from pi
        valid = tf.reduce_max(tf.cast(X, tf.float32), axis = 1) 
        valid_flat = tf.reshape(valid, [-1, nact]) # this is the equavalent of "ind_flat_filter"
        pi_fil = pi + (valid_flat - tf.ones(tf.shape(valid_flat))) * 1e32
    return pi_fil, vf[:, 0]