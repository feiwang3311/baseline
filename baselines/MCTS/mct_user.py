import os, time, pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils import discount_with_dones, Scheduler, make_path, find_trainable_variables, cat_entropy, mse, conv, fc, conv_to_fc
from models import model2
import scipy.sparse as sp
from sl_buffer import slBuffer
from mct import MCT

def load(params, load_path):
    load_file = os.path.join(load_path, "saved")
    with open(load_file, "rb") as fileToload:
        loaded_params = pickle.load(fileToload)
    restores = []
    for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
    return restores

def save(ps, save_path):
    os.makedirs(save_path, exist_ok = True)
    save_file = os.path.join(save_path, "saved")
    with open(save_file, "wb") as fileToSave:
        pickle.dump(ps, fileToSave)

"""
    c_act (exploration parameter of MCTS) and num_mcts (the full size of MCTS tree) are determined in minisat.core.Const.h
    NOTE: max_clause, max_var and nc are define in both here (for the model) and in minisat.core.Const.h (for writing states). They need to BE the same.
    nbatch is the degree of parallel            (defautl 1) TODO: may increase this for efficience
    nstack is the number of history for a state (default 1)
"""
def self_play(args, scope, nrepeat, nbatch = 2, nstack = 1):
    # basic setup of env, model, and directory for saving trace (state, PI, num_step)
    os.makedirs(args.dump_dir, exist_ok = True)
    dump_trace = os.path.join(args.dump_dir, args.dump_file)
    MCTList = []
    for i in range(nbatch):
        MCTList.append(MCT(args.train_path, i, args.max_clause, args.max_var, nrepeat))
    nh = args.max_clause
    nw = args.max_var
    nc = 2
    nact = 2 * nw
    ob_shape = (nbatch, nh, nw, nc * nstack)
    X = tf.placeholder(tf.float32, ob_shape)
    p, v = model2(X, nact, scope)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        params = find_trainable_variables(scope)
        if (args.save_dir is not None) and (args.model_dir is not None):
            sess.run(load(params, os.path.join(args.save_dir, args.model_dir)))
            print("loaded model {} at dir {}".format(args.save_dir, args.model_dir))
        else:
            # this is the initial random parameter, let's save it here
            ps = sess.run(params)
            save(ps, os.path.join(args.save_dir, "model-0"))

        # accumulate states from MCTList, evaluate them in neural net, call get_state again with pi and v as parameters, stop of all states are None
        pi_matrix = np.zeros((nbatch, nact), dtype = np.float32)
        v_array = np.zeros((nbatch,), dtype = np.float32)
        needMore = np.ones((nbatch,), dtype = np.bool)
        dummy = []
        for _ in range(nbatch):
        	dummy.append(np.zeros((nh, nw, nc), dtype = np.float32))
        while np.any(needMore):
            states = []
            for i in range(nbatch):
            	if needMore[i]: 
            		temp = MCTList[i].get_state(pi_matrix[i], v_array[i])
            	if (not needMore[i]) or (temp is None):
            		needMore[i] = False
            		states.append(dummy[i])
            	else:
                    states.append(temp)
            pi_matrix, v_array = sess.run([p, v], feed_dict = {X: np.asarray(states, dtype = np.float32)}) 

        print("loop finished and save Pi graph to slBuffer")
        if os.path.isfile(dump_trace):
            with open(dump_trace, 'rb') as sl_file:
                sl_Buffer = pickle.load(sl_file)
        else:
            sl_Buffer = slBuffer(args.sl_buffer_size)     
        for i in range(nbatch):
            MCTList[i].write_data_to_buffer(sl_Buffer)
        with open(dump_trace, 'wb') as sl_file:
            pickle.dump(sl_Buffer, sl_file, -1)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # currently these arguments are useless
    parser.add_argument('--env_id', help='environment ID', default='gym_sat_Env-v0') 
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'dnn'], default='cnn') # Change for SAT: use dnn as default
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=1) # Change for SAT: use 1, was 40
    
    # actually useful arguments are here
    parser.add_argument('--save_dir', type = str, help='where is the model saved', default="parameters/")
    parser.add_argument('--model_dir', type = str, help='which model to test on', default="model-0")
    parser.add_argument('--train_path', type = str, help='where are training files', default="graph_train/")
    parser.add_argument('--test_path', type = str, help='where are test files', default="graph_test/")
    parser.add_argument('--max_clause', type = int, help="what is the max_clause", default=100)
    parser.add_argument('--max_var', type = int, help="what is the max_var", default=20)
    parser.add_argument('--train_mode', type = str, help="choose random, iterate, repeat^n, filename", default="iterate")
    parser.add_argument('--dump_dir', type = str, help="where to save (state, Pi, num_step) for SL", default = "SLRaw")
    parser.add_argument('--dump_file', type = str, help="what is the filename to save (state, Pi, num_step) for SL", default="sl.pkl")
    parser.add_argument('--sl_buffer_size', type = int, help="max size of sl buffer", default = 1000000)
    parser.add_argument('--nbatch', type = int, help="what is the batch size to use", default = 20)
    parser.add_argument('--nrepeat', type = int, help="how many times to repeat a SAT problem", default=200)

    args = parser.parse_args()
    self_play(args, scope="mcts", nrepeat = args.nrepeat, nbatch = args.nbatch)

if __name__ == '__main__':
    main()
