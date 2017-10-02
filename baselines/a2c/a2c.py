import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.policies import CnnPolicy
from baselines.a2c.utils import cat_entropy, mse

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        writter = tf.summary.FileWriter("/tmp/a2c_demo/1") # Change for SAT: this is to use tensorBoard


        A = tf.placeholder(tf.int32, [nbatch]) # Comments by Fei: this must be the action
        ADV = tf.placeholder(tf.float32, [nbatch]) # Comments by Fei: this must be the advantage 
        R = tf.placeholder(tf.float32, [nbatch]) # Comments by Fei: this must be the reward
        LR = tf.placeholder(tf.float32, []) # Comments by Fei: this must be the learning rate

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A) # Comments by Fei: pi is nbatch * nact
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            # writter.add_graph(sess.graph)
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            # make_path(save_path) Comments by Fei: this seems to be a bug. joblib.dump cannot write to directory, but make_path made "save_path" a dir
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        """
            Fei adds function: supervised training
        """
        def super_train():
            # "training is the filename taht we preprocessed our training data"
            filename = "training"
            print("read files from snaps, temporally save it in %s" % (filename))
            from preprocess import data_shuffler
            data = data_shuffler(filename)
            [num_train, Xlen] = data.train["trainX"].shape
            num_var = int(data.max_var)
            num_clause = int(data.max_clause)
            # X is Xlen array (which needs reshape before training), Y is nact array
            y_ = tf.placeholder(tf.float32, shape = [None, 2 * num_var], name = "labels")
            # Train and evaluate
            with tf.name_scope("loss"):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = self.train_model.pi))
            with tf.name_scope("train"):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            with tf.name_scope("accuracy"):
                correct_prediction = tf.equal(tf.argmax(self.train_model.pi, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # also need to supervisely train the step_model network
            with tf.name_scope("loss_step"):
                cross_entropy_step = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = self.step_model.pi))
            with tf.name_scope("train_step"):
                train_step_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_step)
            with tf.name_scope("accuracy_step"):
                correct_prediction_step = tf.equal(tf.argmax(self.step_model.pi, 1), tf.argmax(y_, 1))
                accuracy_step = tf.reduce_mean(tf.cast(correct_prediction_step, tf.float32))
            
            sess.run(tf.global_variables_initializer())
            # supervised training cycle for step_model
            for i in range(2001):
                batch = data.next_batch(nenvs)
                x_reshape = np.reshape(batch[0], [-1, num_clause, num_var, 1])
                feed_dict = {step_model.X: x_reshape, y_: batch[1]}
                if i % 500 == 0:
                    train_accuracy = sess.run(accuracy_step, feed_dict)
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                sess.run(train_step_step, feed_dict)
            # supervised testing cycle
            test_x_reshape = np.reshape(data.test["testX"], [-1, num_clause, num_var, 1])
            num_round_step = round(test_x_reshape.shape[0] / nenvs) - 1
            for i in range(num_round_step):
                feed_dict = {step_model.X: test_x_reshape[i * nenvs: (i+1)*nenvs], y_: data.test["testY"][i*nenvs: (i+1)*nenvs]}
                print('test accuracy %g' % sess.run(accuracy_step, feed_dict))
            # supervised training cycle
            for i in range(201):
                batch = data.next_batch(nbatch) # TODO: need np reshape, not tf reshape
                x_reshape = np.reshape(batch[0], [-1, num_clause, num_var, 1])
                feed_dict={train_model.X: x_reshape, y_: batch[1]}
                if i % 100 == 0:                     
                    train_accuracy = sess.run(accuracy, feed_dict)
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                sess.run(train_step, feed_dict)
            # supervised testing cycle
            num_round = round(test_x_reshape.shape[0] / nbatch) - 1
            for i in range(num_round):
                feed_dict = {train_model.X: test_x_reshape[i*nbatch:(i+1)*nbatch], y_: data.test["testY"][i*nbatch:(i+1)*nbatch]}
                print('test accuracy %g' % sess.run(accuracy, feed_dict))

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        self.super_train = super_train
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.int8) # Comments by Fei: be aware, self.obs is only nenv! Changes for SAT: use int8 (not uint8)
        obs = env.reset() # Comments by Fei: obs should only be nenv * nh * nw * 1. State may contain several obs in a roll, by nstack.
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs
        # self.obs[:, :, :, -1] = obs[:, :, :, 0] Change for SAT

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones) # Comments by Fei: step_model (nstep = 1)! nenv, nenv, nenv * 2nlstm
            mb_obs.append(np.copy(self.obs)) # Comments by Fei: finally will be nsteps * nenv * nh * nw * (nc*nstack)
            mb_actions.append(actions) # Comments by Fei: finally will be nsteps * nenv
            mb_values.append(values) # Comments by Fei: finally will be nsteps * nenv
            mb_dones.append(self.dones) 
            obs, rewards, dones, _ = self.env.step(actions) # Comments by Fei: nenv * nh * nw * 1, nenv, nenv
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            mb_rewards.append(rewards) # Comments by Fei: finally will be nsteps * nenv
        mb_dones.append(self.dones) # Comments by Fei: finally will be (nsteps+1) * nenv
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape) # Comments by Fei: (nenv*nsteps, nh, nw, nc*nstack)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0) # Comments by Fei: nenv * nsteps
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0) # Comments by Fei: nenv * nsteps
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0) # Comments by Fei: nenv * nsteps
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0) # Comments by Fei: nenv * (nsteps+1)
        mb_masks = mb_dones[:, :-1] # Comments by Fei: masks is nenv * nsteps (missing the last done)
        mb_dones = mb_dones[:, 1:] # Comments by Fei: dones is nenv * nsteps (missing the first done)
        last_values = self.model.value(self.obs, self.states, self.dones).tolist() # Comments by Fei: step_model (nstep = 1)! nenv vector
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)): # Comments by Fei: nenv | nsteps, nsteps, 1
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten() # Comments by Fei: nbatch vector now
        mb_actions = mb_actions.flatten()# Comments by Fei: nbatch vector now
        mb_values = mb_values.flatten()# Comments by Fei: nbatch vector now
        mb_masks = mb_masks.flatten()# Comments by Fei: nbatch vector now
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

    """
        this function play the game of SAT and return the performance (Added by Fei)
    """
    def play(self, decision = 'play'):
        obs = self.env.reset()
        states = self.model.initial_state
        nenv = self.env.num_envs
        dones = [False for _ in range(nenv)]
        mask = np.asarray([1.0 for _ in range(nenv)])
        sum_rewards = np.zeros(nenv)
        for n in range(self.nsteps * 10):
            if decision == 'play':
                actions, values, states = self.model.step(np.expand_dims(obs, 3), states, dones)
            else: 
                actions = [-1 for _ in range(nenv)] # use -1 as inidcation of default choice by SAT solver
            obs, rewards, dones, _ = self.env.step(actions)
            # for those environments that have not be done once, accumulate scores
            mask[np.asarray(dones)] = 0.0
            sum_rewards = sum_rewards + np.asarray(rewards) * mask
            if not mask.any(): break
        return np.mean(sum_rewards)

# Change for SAT, nstack changed to 1 (was 4), nsteps changed to 20, was 5
def learn(policy, env, seed, nsteps=20, nstack=1, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    # add supervised learning here:
    # model.super_train()

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run() # Comments by Fei: (nenv*nsteps, nh, nw, nc*nstack), (nenv, nlstm*2), nbatch, nbatch, nbatch, nbatch
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            #logger.record_tabular("play performance", float(runner.play()))
            #logger.record_tabular("default performance", float(runner.play(decision = "minisat")))
            logger.dump_tabular()
            model.save("params" + str(update // log_interval // 10))
    env.close()

if __name__ == '__main__':
    main()
