import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
import gym

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps) # Comments by Fei: xs is list of nsteps, each is nenv * nh
            ms = batch_to_seq(M, nenv, nsteps) # Comments by Fei: ms is list of nsteps, each is nenv vector
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm) # Comment by Fei: h5 is the same dimension as xs, but with value changed by LSTM. snew is new S
            h5 = seq_to_batch(h5) # Comments by Fei: h5 is nbatch * nh again, just like h4
            pi = fc(h5, 'pi', nact, act=lambda x:x) # Comments by Fei: pi is nbatch * nact
            vf = fc(h5, 'v', 1, act=lambda x:x) # Comments by Fei: vf is nbatch * 1

        v0 = vf[:, 0] # Comments by Fei: v0 is nbatch vector, each value is the value function of a state
        a0 = sample(pi) # Comments by Fei: a0 is nbatch vector, each value is the best choice of action, at that state
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.int8, ob_shape) #obs Change for SAT: SAT input is of type int8!
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32), 'c1', nf=32, rf=8, stride=1, init_scale=np.sqrt(2)) # Change for SAT, don't divide input by 255.
            # h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)) # Change for SAT, don't divide input by 255.
            h2 = conv(h, 'c2', nf=64, rf=4, stride=1, init_scale=np.sqrt(2)) # Change for SAT. stride was 2
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)) 
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        # Comments by Fei: filter invalid actions!
        pos = tf.reduce_max(X, axis = 1) # Comments by Fei: get 1 if the postive variable exists in any clauses, otherwise 0
        neg = tf.reduce_min(X, axis = 1) # Comments by Fei: get -1 if the negative variables exists in any clauses, otherwise 0
        ind = tf.concat([pos, neg], axis = 2) # Comments by Fei: get (1, -1) if this var is present, (1, 0) if only as positive, (0, -1) if only as negative
        ind_flat = tf.reshape(ind, [-1, nw * 2]) # Comments by Fei: this is nbatch * nact, with 0 values labeling non_valid actions, 1 or -1 for other
        ind_flat_filter = tf.abs(tf.cast(ind_flat, tf.float32)) # Comments by Fei: this is nbatch * nact, with 0 values labeling non_valid actions, 1 for other
        pi_min = tf.reduce_min(pi, axis = 1)
        pi_adjust = pi - tf.expand_dims(pi_min, axis = 1) # Comments by Fei: make sure the maximal values are positive
        pi_filter = pi_adjust * ind_flat_filter # Comments by Fei: zero-fy non-valid values, don't change valid values
        pi_filter_adjust = pi_filter + tf.expand_dims(pi_min, axis = 1) # Comments by Fei: adjust back (to keep the valid values unchanged)
        a0 = sample(pi_filter_adjust)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class DnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse = False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc* nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.int8, ob_shape)
        with tf.variable_scope("model", reuse = reuse):
            x = conv_to_fc(tf.cast(X, tf.float32))
            A1 = fc(x, 'fc1', nh = int(nh * nw), init_scale = 0.1)
            A2 = fc(A1, 'fc2', nh = int(nh * nw), init_scale = 0.1)
            A4 = fc(A2, 'fc3', nh = int(nh * nw / 4), init_scale = 0.1)
            A5 = fc(A4, 'fc4', nh = int(nh * nw / 4), init_scale = 0.1)
            A7 = fc(A5, 'fc5', nh = int(nh * nw / 10), init_scale = 0.1)
            A8 = fc(A7, 'fc6', nh = int(nh * nw / 10), init_scale = 0.1)
    
            pi = fc(A8, 'fc7', nh = 2*nw, act = lambda x:x, init_scale = 0.1)
            vf = fc(A8, 'v', 1, act = lambda x:x, init_scale = 0.1)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] # not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v, []
        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})
        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
