import os, logging, gym, time, pickle
import numpy as np
import tensorflow as tf
import math
import tensorflow.contrib.layers as layers
from baselines import logger
from baselines.MCTS.utils import discount_with_dones, Scheduler, make_path, find_trainable_variables, cat_entropy, mse, conv, fc, conv_to_fc
from models import model2

def load(params, load_path):
    with open(load_path, "rb") as file:
        loaded_params = pickle.load(file)
    restores = []
    for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
    return restores

def save(ps, save_path):
    with open(save_path, "wb") as file:
        pickle.dump(ps, file)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_tau(num_step):
    return 0.9 # TODO: decide how to generate tau based on steps

from multiprocessing import Pipe, Process
import scipy.sparse as sp
from minisat.minisat.gym.GymSolver import minisat_wrapper
"""
    cact is the exploration parameter of MCTS
    nbatch is the degree of parallel
    num_mcts is the max size of MCTS tree 
    nstack is the number of history for a state
"""
def self_play(args, scope, cact, num_mcts, nrepeat, nbatch = 1, nstack = 1):
    # basic setup of env and model 
    minisat = minisat_wrapper(args.train_path, max_clause = args.max_clause, max_var = args.max_var, mode = args.train_mode)
    nh, nw, nc = minisat.observation_space.shape
    nact = minisat.action_space
    ob_shape = (1, nh, nw, nc * 1)
    X = tf.placeholder(tf.bool, ob_shape)
    p, v = model2(X, nact, scope)

    # set up tensorflow server by forking a server process and a main process
    pp, cp = Pipe()
    pip = os.fork()
    if not pip:
        # server thread
        cp.close()
        server = tf.train.Server.create_local_server(start = True)
        pp.send(server.target)
        with tf.Session(server.target) as sess:
            sess.run(tf.global_variables_initializer())
            params = find_trainable_variables(scope)
            if (args.save_dir is not None) and (args.model_dir is not None):
                sess.run(load(params, os.path.join(args.save_dir, args.model_dir, "save")))
                print("loaded model {} at dir {}".format(args.save_dir, args.model_dir))
            else:
                # this is the initial random parameter, let's save it here
                ps = sess.run(params)
                save(ps, os.path.join(args.save_dir, "model-0", "save"))

        minisat = None
        pp.close()
        # then this thread must not pre-terminate before any other threads that use server!
        while True:
            time.sleep(1000)
        return

    # main thread and basic setups
    pp.close()
    target = cp.recv()
    cp.close()
    os.makedirs(args.dump_dir, exist_ok = True)
    dump_trace = os.path.join(args.dump_dir, args.dump_file) # this is the file to which I write the play trace (state, Pi, num_step)

    num_steps = nrepeat * minisat.sat_file_num
    for _ in range(num_steps): # solve nrepeat times for the given environment 
        # initialization of important explore parameters before entering the MCTS cycle
        num_step = 0
        state = minisat.reset()
        with tf.Session(target) as sess:
            pi, v0 = sess.run([p, v], feed_dict = {X: state[None]})
        P = softmax(pi)
        Q = np.zeros(nact)
        W = np.zeros(nact)
        N = np.zeros(nact)
        isdone = False
        pipe_up = None
        pipe_down = {}
        listen = -1 # if listen is -1, listen to the pipe_up, if some non-neg num, listen to pipe_down[listen]
        
        """ For some weird reason this function is not working (variables are not in scope)
        """ 
        def simulate_down_maybe_fork():
            U = cact * P * np.sqrt(np.sum(N) + 1) / (1 + N)
            explore = np.argmax(Q + U)
            if explore in pipe_down: # send the explore signal to pipe_down[explore]
                pipe_down[explore].send(-1) # Note: parent send -1 mean to explore!
                listen = explore
            else: # need fork a shadow clone, let the child process step it
                p1, p2 = Pipe()
                pid = os.fork()
                if pid: # Parent, use p1
                    p2.close()
                    listen = explore
                    pipe_down[explore] = p1
                else: # Child, use p2, make step (this is new leaf node)
                    p1.close()
                    pipe_up = p2
                    pipe_down.clear()
                    listen = -1
                    state, _, done, _ = minisat.step(explore)
                    num_step += 1
                    if done:
                        isdone = True
                        pipe_up.send(1.0)
                    else:
                        with tf.Session(target) as sess1:
                            pi, v0 = sess1.run([p, v], feed_dict = {X: state[None]})
                        P = softmax(pi)
                        Q = np.zeros(nact)
                        W = np.zeros(nact)
                        N = np.zeros(nact)
                        pipe_up.send(np.asscalar(v0[0])) 

        # main MCTS cycle by fork()
        while True:
            
            if (listen >= 0) and (not isdone): # case of listen to child and not done
                assert listen in pipe_down, "listening to a child pipe {} that does not exists".format(listen)
                inputs = pipe_down[listen].recv() # wait for the information from the child you are listening to
                assert isinstance(inputs, float), "pipe message from child {} is not float, but {}".format(listen, inputs)
                N[listen] += 1.0
                W[listen] += inputs
                Q[listen] = W[listen] / N[listen]
                listen = -1
                if pipe_up is not None:
                    pipe_up.send(inputs)

            elif (listen >= 0) and (isdone): # case of listen to child and done ERROR
                assert False, "ERROR!! listening to child but state is done (no children)"

            elif (listen == -1) and (pipe_up is None): # in the root of simulation tree, initiation step
                if isdone: # the root node is in a finished state, just break, so that we go back to the for loop
                    break
                elif np.sum(N) < num_mcts: # more simulation to be done
                    U = cact * P * np.sqrt(np.sum(N) + 1) / (1 + N)
                    explore = np.argmax(Q + U)
                    if explore in pipe_down: # send the explore signal to pipe_down[explore]
                        pipe_down[explore].send(-1) # Note: parent send -1 mean to explore!
                        listen = explore
                    else: # need fork a shadow clone, let the child process step it
                        p1, p2 = Pipe()
                        pid = os.fork()
                        if pid: # Parent, use p1
                            p2.close()
                            listen = explore
                            pipe_down[explore] = p1
                        else: # Child, use p2, make step (this is new leaf node)
                            p1.close()
                            pipe_up = p2
                            pipe_down.clear()
                            listen = -1
                            state, _, done, _ = minisat.step(explore)
                            num_step += 1
                            if done:
                                # no need to run model, what to send up?
                                isdone = True
                                pipe_up.send(1.0)
                            else:
                                sess = tf.Session(target)
                                pi, v0 = sess.run([p, v], feed_dict = {X: state[None]})
                                sess.close()
                                P = softmax(pi)
                                Q = np.zeros(nact)
                                W = np.zeros(nact)
                                N = np.zeros(nact)
                                pipe_up.send(np.asscalar(v0[0]))
                else: # finished all simulation
                    # get Pi and save it to file
                    print(num_step)
                    tau = get_tau(num_step)
                    Pi = np.power(N, 1/tau)
                    Pi = Pi / np.sum(Pi)
                    with open(dump_trace, "ab") as append_here:
                        assert len(state.shape) == 3, "state to save should be a 3D sparse matrix"
                        sparse_list = []
                        for layer in range(state.shape[2]):
                            sparse_list.append(sp.csc_matrix(state[:,:,layer]))
                        pickle.dump((sparse_list, Pi, num_step), append_here) # write the (state, Pi, num_step) to file dump_trace
                    # actually take this step
                    action = np.random.choice(range(nact), p = Pi)
                    assert action in pipe_down, "action of choice {} does not exists".format(action)
                    for branch in pipe_down: # ask all non action simulation processes to terminate (watch out for EOFERROR)
                        if branch != action:
                            pipe_down[branch].send(-2)
                            pipe_down[branch].close()
                    pipe_down[action].send(-3) # ask action simulation process to take the lead as new root
                    pipe_down[action].close()
                    return # terminate myself process

            elif (listen == -1) and (pipe_up is not None):
                inputs = pipe_up.recv()
                assert isinstance(inputs, int), "pipe message from parent is not int, but {}".format(inputs)
                if inputs == -1: # means to go on and simulate
                    if isdone:
                        pipe_up.send(1.0)
                    else:
                        U = cact * P * np.sqrt(np.sum(N) + 1) / (1 + N)
                        explore = np.argmax(Q + U)
                        if explore in pipe_down: # send the explore signal to pipe_down[explore]
                            pipe_down[explore].send(-1) # Note: parent send -1 mean to explore!
                            listen = explore
                        else: # need fork a shadow clone, let the child process step it
                            p1, p2 = Pipe()
                            pid = os.fork()
                            if pid: # Parent, use p1
                                p2.close()
                                listen = explore
                                pipe_down[explore] = p1
                            else: # Child, use p2, make step (this is new leaf node)
                                p1.close()
                                pipe_up = p2
                                pipe_down.clear()
                                listen = -1
                                state, _, done, _ = minisat.step(explore)
                                num_step += 1
                                if done:
                                    # no need to run model, what to send up?
                                    isdone = True
                                    pipe_up.send(1.0)
                                else:
                                    sess = tf.Session(target)
                                    pi, v0 = sess.run([p, v], feed_dict = {X: state[None]})
                                    sess.close()
                                    P = softmax(pi)
                                    Q = np.zeros(nact)
                                    W = np.zeros(nact)
                                    N = np.zeros(nact)
                                    pipe_up.send(np.asscalar(v0[0]))
                elif inputs == -2: # means to terminate (unused simulation branch)
                    pipe_up.close()
                    assert (not isdone) or (len(pipe_down) == 0), "ill state, isdone but still have children {}".format(pipe_down)
                    for branch in pipe_down:
                        pipe_down[branch].send(-2)
                        pipe_down[branch].close()
                    return
                elif inputs == -3: # means to take over as new root
                    pipe_up.close()
                    pipe_up = None
                else:
                    assert False, "unexpected message from pipe_up {}".format(inputs)
            
    # after running it nrepeat time, signal server to terminate
    import signal
    os.kill(pid, signal.SIGTERM)
                    
def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='gym_sat_Env-v0') 
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'dnn'], default='cnn') # Change for SAT: use dnn as default
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=1) # Change for SAT: use 1, was 40
    
    # actually useful arguments are here
    parser.add_argument('--save_dir', type = str, help='where is the model saved', default="parameters/")
    parser.add_argument('--model_dir', type = str, help='which model to test on', default='model-0')
    parser.add_argument('--train_path', type = str, help='where are training files', default="graph_train/")
    parser.add_argument('--test_path', type = str, help='where are test files', default="graph_test/")
    parser.add_argument('--max_clause', type = int, help="what is the max_clause", default=300)
    parser.add_argument('--max_var', type = int, help="what is the max_var", default=90)
    parser.add_argument('--train_mode', type = str, help="choose random, iterate, repeat^n, filename", default="iterate")
    parser.add_argument('--dump_dir', help="where to save (state, Pi, num_step) for SL", default = "SLRaw")
    parser.add_argument('--dump_file', help="what is the filename to save (state, Pi, num_step) for SL", default="saveSL")

    parser.add_argument('--num_mcts', type = int, help="what is the size for each MCTS tree", default=20)
    parser.add_argument('--nrepeat', type = int, help="how many times to repeat a SAT problem", default=20)
    parser.add_argument('--cact', type = float, help="what is the hypoparameter for exploration in MCTS", default=1.0)

    args = parser.parse_args()
    self_play(args, scope="mcts", cact = args.cact, num_mcts = args.num_mcts, nrepeat = args.nrepeat)

if __name__ == '__main__':
    main()