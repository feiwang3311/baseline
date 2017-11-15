import os, time, pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils import discount_with_dones, Scheduler, make_path, find_trainable_variables, cat_entropy, mse, conv, fc, conv_to_fc
from models import model2
import scipy.sparse as sp
from sl_buffer import slBuffer
from mct import MCT
from mct_user import load, save

def super_train(args):
    nh = args.max_clause
    nw = args.max_var
    nc = 2
    nact = 2 * nw
    ob_shape = (None, nh, nw, nc * nstack)
    X = tf.placeholder(tf.float32, ob_shape)
    Y = tf.placeholder(tf.float32, (None, nact))
    Z = tf.placeholder(tf.float32, (None))
    
    # model and loss 
    p, v = model2(X, nact, scope)  # Note!! p is not yet passed into softmax function
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=p))
        value_loss = mean_squared_error(labels = Z, predictions = v)
        l2_loss = tf.nn.l2_loss(params)
        loss = cross_entropy + value_loss + args.l2_coeff * l2_loss
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        params = find_trainable_variables(scope)
        assert (args.save_dir is not None) and (args.model_dir is not None), "save_dir and model_dir needs to be specified for super_training"
        sess.run(load(params, os.path.join(args.save_dir, args.model_dir)))
        print("loaded model {} at dir {}".format(args.save_dir, args.model_dir))

        # data for supervised training
        dump_trace = os.path.join(args.dump_dir, args.dump_file)
        with open(dump_trace, 'rb') as sl_file:
            sl_Buffer = pickle.load(sl_file)

        # supervised training cycle
        for i in range(args.sl_num_steps + 1):
            batch = sl_Buffer.sample(args.sl_nbatch)
            feed_dict = { X: batch[0], Y: batch[1], Z: batch[2] }
            sess.run(train_step, feed_dict)
            if i > 0 and i % args.sl_ncheckpoint == 0: # checkpoint the model
                ps = sess.run(params)
                save(ps, os.path.join(args.save_dir, "model-" + ???))
