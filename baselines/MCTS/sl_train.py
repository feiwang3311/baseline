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


def super_train(args, scope, nstack = 1):
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
    params = find_trainable_variables(scope)
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=p))
        value_loss = tf.losses.mean_squared_error(labels = Z, predictions = v)
        lossL2 = tf.add_n([ tf.nn.l2_loss(vv) for vv in params ])
        loss = cross_entropy + value_loss + args.l2_coeff * lossL2
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # load the best model so far ?? 
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
            if i > 0 and i % args.sl_ncheckpoint == 0: 
                model_num = int(args.last_model_num + i // args.sl_ncheckpoint)
                print("checkpoint model {}".format(model_num))
                ps = sess.run(params)
                save(ps, os.path.join(args.save_dir, "model-" + str(model_num)))

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # actually useful arguments are here
    parser.add_argument('--save_dir', type = str, help='where is the model saved', default="parameters/")
    parser.add_argument('--model_dir', type = str, help='which model to load', default="model-0")
    parser.add_argument('--train_path', type = str, help='where are training files', default="graph_train/")
    parser.add_argument('--test_path', type = str, help='where are test files', default="graph_test/")
    parser.add_argument('--max_clause', type = int, help="what is the max_clause", default=100)
    parser.add_argument('--max_var', type = int, help="what is the max_var", default=20)
    parser.add_argument('--train_mode', type = str, help="choose random, iterate, repeat^n, filename", default="iterate")
    parser.add_argument('--dump_dir', type = str, help="where to save (state, Pi, num_step) for SL", default = "SLRaw")
    parser.add_argument('--dump_file', type = str, help="what is the filename to save (state, Pi, num_step) for SL", default="sl.pkl")
    parser.add_argument('--sl_buffer_size', type = int, help="max size of sl buffer", default = 1000000)
    parser.add_argument('--nbatch', type = int, help="what is the batch size to use", default = 32)
    parser.add_argument('--nrepeat', type = int, help="how many times to repeat a SAT problem", default=200)

    # supervised learning specific arguments
    parser.add_argument('--l2_coeff', type = float, help="the coefficient for l2 regularization", default = 0.0001)
    parser.add_argument('--sl_num_steps', type = int, help="how many times to do supervised training", default = 10000)
    parser.add_argument('--sl_nbatch', type = int, help="what is the batch size for supervised training", default = 32)
    parser.add_argument('--sl_ncheckpoint', type = int, help="how often to checkpoint a supervised trained model", default = 10)
    parser.add_argument('--last_model_num', type = int, help="what is the last model number we saved", default = 0)

    args = parser.parse_args()
    super_train(args, scope = "supervised")

if __name__ == '__main__':
    main()