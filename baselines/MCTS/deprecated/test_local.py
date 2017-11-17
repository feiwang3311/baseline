import os
import tensorflow as tf
import numpy as np
from multiprocessing import Pipe, Process
from baselines.MCTS.utils import discount_with_dones, Scheduler, make_path, find_trainable_variables, cat_entropy, mse, conv, fc, conv_to_fc

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
        pi_min = tf.reduce_min(pi, axis = 1)
        pi_adjust = pi - tf.expand_dims(pi_min, axis = 1) # Comments by Fei: make sure the maximal values are positive
        pi_filter = pi_adjust * ind_flat_filter # Comments by Fei: zero-fy non-valid values, don't change valid values
        pi_filter_adjust = pi_filter + tf.expand_dims(pi_min, axis = 1) # Comments by Fei: adjust back (to keep the valid values unchanged) TODO: maybe not necessary
    
    return pi_filter_adjust, vf[:, 0]

def func():
  # basic setup of model
  import gym
  env = gym.make("gym_sat_Env-v0")
  nh, nw, nc = env.observation_space.shape
  nact = env.action_space.n
  ob_shape = (1, nh, nw, nc*1)

  X = tf.placeholder(tf.int8, ob_shape) # TODO: SAT may use boolean type
  p, v = model(X, nact, "scope")

  # fork, so that one branch set up the server, the other use the sessiong
  pp, cp = Pipe()
  pip = os.fork()
  if not pip:
    # server
    cp.close()
    server = tf.train.Server.create_local_server()
    pp.send(server.target)
    with tf.Session(server.target) as sess:
      sess.run(tf.global_variables_initializer())
      c = tf.constant("server initialized parameters")
      print(sess.run(c))
    pp.close()
    return

  else:
    # main thread
    pp.close()
    inputs = cp.recv()
    cp.close()
    print("target is {}".format(inputs))
    
    with tf.Session(inputs) as sess:
      c = tf.constant("child tensorflow works too")
      print(sess.run(c))
      state = env.reset()
      pi, v0 = sess.run([p, v], feed_dict = {X: state[None]})
      print(pi)
      print(v0)
      vv = v0[0]
      vv_sc = np.asscalar(vv)
    
    


func()

exit(0)
def set_up_server(conn):
    server = tf.train.Server.create_local_server()
    with tf.Session(server.target) as sess:
      print(sess.run(tf.constant("from creator")))
    conn.send(server.target)
    conn.close()

parent_conn, child_conn = Pipe()
p = Process(target=set_up_server, args=(child_conn,))
p.start()
target = parent_conn.recv()  
print("found target as {}".format(target))
p.join()
with tf.Session(target) as sess:
  print(sess.run(tf.constant("from child")))

exit(0)