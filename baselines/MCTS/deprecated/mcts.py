import os, time, pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils import discount_with_dones, Scheduler, make_path, find_trainable_variables, cat_entropy, mse, conv, fc, conv_to_fc
from models import model2
import scipy.sparse as sp
from minisat.minisat.gym.GymSolver import sat
from sl_buffer import slBuffer

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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_PI(counts, tau):
    Pi  = np.power(counts, 1/tau)
    Pi = Pi / np.sum(Pi)
    return Pi

def is_step_valid_in_state(state, step): # NO LONGER USED
    # state is max_clause * max_var * 2 and step is in range(max_var*2)
    # this function returns true if step is valid in state ()
    valid = np.any(state, axis = 0) # valid is now max_var * 2
    [max_var, nc] = valid.shape     # 
    valid_flat = np.reshape(valid, [max_var * nc,])
    return valid_flat[step] > 0.5

def get_nrepeat_count(action, nact):
    # given a numpy array of action, return an numpy array of the size of nact, and contains the counts of each element
    action = np.sort(action)
    counts = np.zeros(nact, dtype = int)
    base = 0
    start = 0
    for i in range(action.shape[0]):
        if action[i] > base:
            counts[base] = i - start
            start = i
            base = action[i]
    counts[base] = action.shape[0] - start
    print(counts)
    return counts

class Pi_struct(object):
    def __init__(self, size, repeat, level, parent = None):
        self.size = size
        self.repeat = repeat
        self.level = level
        self.children = {}
        self.next = 0
        self.evaluated = False
        self.parent = parent
        self.score = 0
    def add_state(self, state):
        self.isValid = np.reshape(np.any(state, axis = 0), [self.size,])
        state_2d = np.reshape(state, [-1, state.shape[1] * state.shape[2]])
        self.state = sp.csc_matrix(state_2d)
    def add_Pi(self, Pi):
        self.Pi = np.copy(Pi)
        action = np.random.choice(range(self.size), self.repeat, p = Pi) # random select actions based on Pi, for nrepeat times
        self.nrepeats = get_nrepeat_count(action, self.size)
        for i in range(self.size):
            if self.nrepeats[i] > 0.5:
                self.children[i] = Pi_struct(self.size, self.nrepeats[i], self.level+1, parent = self)
        self.evaluated = True
    def get_next(self):
        while self.next < self.size and (self.nrepeats[self.next] < 0.5 or not self.isValid[self.next]):
            self.next += 1
        if self.next >= self.size: 
            return -1 # no more actions to go
        return self.next
    def set_next(self, additional_score):
        # print("node at level {} added score of worth {}".format(self.level, additional_score))
        self.score += additional_score
        self.next += 1
        next = self.get_next()
        if (self.get_next() < 0 and self.parent is not None):
            return self.parent.set_next(self.score)
        return next

def analyze_Pi_graph_dump(Pi_node, sl_Buffer, standard):
    if not Pi_node.evaluated: return # this is the finished node (no state, not evaluated)
    for act in Pi_node.children:
        analyze_Pi_graph_dump(Pi_node.children[act], sl_Buffer, standard)
    # save this node's infor TODO: should the score be more quatified?
    av = Pi_node.score / Pi_node.repeat 
    if av > standard[1]: score = - (av-standard[1]) / (standard[2]-standard[1])
    elif av < standard[1]: score = (standard[1]-av) / (standard[1]-standard[0])
    else: score = 0
    sl_Buffer.add(Pi_node.state, Pi_node.Pi, score, Pi_node.repeat)
        
"""
    c_act (exploration parameter of MCTS) and num_mcts (the full size of MCTS tree) are determined in minisat.core.Const.h
    NOTE: max_clause, max_var and nc are define in both here (for the model) and in minisat.core.Const.h (for writing states). They need to BE the same.
    nbatch is the degree of parallel            (defautl 1) TODO: may increase this for efficience
    nstack is the number of history for a state (default 1)
"""
def self_play(args, scope, nrepeat, nbatch = 1, nstack = 1):
    # basic setup of env, model, and directory for saving trace (state, PI, num_step)
    os.makedirs(args.dump_dir, exist_ok = True)
    dump_trace = os.path.join(args.dump_dir, args.dump_file)
    env = sat(args.train_path, max_clause = args.max_clause, max_var = args.max_var, mode = args.train_mode)
    nh, nw, nc = env.observation_space.shape
    nact = env.action_space              # should be 2 * nw
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

        for _ in range(env.sat_file_num):
            # initialization 
            Pi_current = Pi_root = Pi_struct(nact, nrepeat, 0)
            state = env.reset()
            Pi_current.add_state(state)
            min_step = 10000000
            max_step = 0
            
            while True:
                if not Pi_current.evaluated:
                    print("simulation -----------------------------------------------{} at level {}".format(Pi_current.repeat, Pi_current.level))
                    needEnv = True
                    while needEnv or needSim:
                    	if needEnv:
                    		pi, v0 = sess.run([p, v], feed_dict = {X: state[None]})
                    	state, needEnv, needSim = env.simulate(softmax(pi[0]), v0[0])
                    counts = env.get_visit_count()
                    Pi = get_PI(counts, 0.9) # TODO: decide how to control tau later
                    Pi_current.add_Pi(Pi)    # this function also establish the nrepeats for this Pi, and set up the children dictionary

                next_act = Pi_current.get_next() 
                assert next_act >= 0, "next_act is neg, but this is actually never happening, because the set_next() method took care of it"
                isDone, state = env.step(next_act) # there is a guarantee that this next_act is valid
                if isDone:
                    print("step {} to DONE+++++++++++++++++++++++++++++++++++++++++++++++++++++".format(next_act))
                    if Pi_current.level < min_step: min_step = Pi_current.level # update score range for this sat prob
                    if Pi_current.level > max_step: max_step = Pi_current.level
                    if (Pi_current.set_next(Pi_current.level * Pi_current.nrepeats[next_act]) < 0): # write back the total score from this leaf node
                        break # we are finished
                    Pi_current = Pi_root
                    state = env.reset()
                else:
                    print("step {} is continuing***********************************************".format(next_act))
                    Pi_current = Pi_current.children[next_act]
                    Pi_current.add_state(state)
                
            print("loop finished and save Pi graph to slBuffer")
            # save date in sl_buffer
            if os.path.isfile(dump_trace):
                with open(dump_trace, 'rb') as sl_file:
                    sl_Buffer = pickle.load(sl_file)
            else:
                sl_Buffer = slBuffer(args.sl_buffer_size)     
            analyze_Pi_graph_dump(Pi_root, sl_Buffer, (min_step, Pi_root.score / Pi_root.repeat, max_step))
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

    parser.add_argument('--nrepeat', type = int, help="how many times to repeat a SAT problem", default=1)

    args = parser.parse_args()
    self_play(args, scope="mcts", nrepeat = args.nrepeat)

if __name__ == '__main__':
    main()

"""
next_act = Pi_current.get_next() 
shoud_set_next = False
if next_act < 0: # all acts have been done, let parent set_next()
    shoud_set_next = True
    print("next_act is neg, but this is actually never happening, because the set_next() method took care of it")
else:
    isDone, state = env.step(next_act) # there is a guarantee that this next_act is valid
    Pi_current = Pi_current.children[next_act]
    if isDone:
        print("step {} to DONE+++++++++++++++++++++++++++++++++++++++++++++++++++++".format(next_act))
        shoud_set_next = True
    else:
        print("step {} is continuing***********************************************".format(next_act))
        Pi_current.add_state(state)

if shoud_set_next:
    if (Pi_current.parent is None) or (Pi_current.parent.set_next() < 0):
        break # we are finished
    Pi_current = Pi_root
    state = env.reset()
"""
'''
next_act = Pi_current.get_next() 
assert next_act >= 0, "next_act is neg, but this is actually never happening, because the set_next() method took care of it"
isDone, state = env.step(next_act) # there is a guarantee that this next_act is valid
Pi_current = Pi_current.children[next_act]
if isDone:
    print("step {} to DONE+++++++++++++++++++++++++++++++++++++++++++++++++++++".format(next_act))
    if (Pi_current.parent is None) or (Pi_current.parent.set_next() < 0):
        break # we are finished
    Pi_current = Pi_root
    state = env.reset()
else:
    print("step {} is continuing***********************************************".format(next_act))
    Pi_current.add_state(state)
'''