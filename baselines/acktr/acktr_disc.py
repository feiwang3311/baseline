import os.path as osp
import time
import pickle
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance

from baselines.acktr.utils import discount_with_dones
from baselines.acktr.utils import Scheduler, find_trainable_variables
from baselines.acktr.utils import cat_entropy, mse
from baselines.acktr import kfac


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs,total_timesteps, nprocs=32, nsteps=20,
                 nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch]) # Comments by Fei: this is the array of actions for the batch of states
        ADV = tf.placeholder(tf.float32, [nbatch]) # Comments by Fei: this is the array of advantages for the batch of states
        R = tf.placeholder(tf.float32, [nbatch]) # Comments by Fei: this is the array of rewards for the batch of states
        PG_LR = tf.placeholder(tf.float32, []) # Comments by Fei: learning rate of Policy gradient
        VF_LR = tf.placeholder(tf.float32, []) # Comments by Fei: learning rate of value function

        self.model = step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False) # Comments by Fei: used for roll out
        self.model2 = train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True) # Comments by Fei: same neural net, used for train

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        self.logits = logits = train_model.pi

        ##training loss
        pg_loss = tf.reduce_mean(ADV*logpac)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        train_loss = pg_loss + vf_coef * vf_loss


        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
        sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        self.vf_fisher = vf_fisher_loss = - vf_fisher_coef*tf.reduce_mean(tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params=params = find_trainable_variables("model") # Comments by Fei: Need to find out where "model" is used as variable scope

        self.grads_check = grads = tf.gradients(train_loss,params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(learning_rate=PG_LR, clip_kl=kfac_clip,\
                momentum=0.9, kfac_update=1, epsilon=0.01,\
                stats_decay=0.99, async=1, cold_iter=10, max_grad_norm=max_grad_norm)

            update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads,params)))
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, PG_LR:cur_lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, train_op],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path): # Comments by Fei: changed to pickle dump because cuda and mctesla machine didn't install joblib
            ps = sess.run(params)
            with open(save_path, "wb") as file:
                pickle.dump(ps, file)
            
        def load(load_path): # Comments by Fei: changed to pickle load because cuda and mctesla machine didn't install joblib
            with open(load_path, "rb") as file:
                loaded_params = pickle.load(file)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)


        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps, nstack, gamma):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.int8) # Comments by Fei: be aware, self.obs is only nenv! Changes for SAT: use int8 (not uint8)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs):
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollout
        mb_obs = np.asarray(mb_obs, dtype=np.int8).swapaxes(1, 0).reshape(self.batch_ob_shape) # Comments by Fei: need to change uint8 to int8!!
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

# Change for SAT, nstack changed to 1 (was 4), nsteps changed to 20, was 20, total_timesteps was 40e6
def learn(policy, env, seed, total_timesteps=int(1e6), gamma=0.99, log_interval=100, nprocs=32, nsteps=20,
                 nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, save_interval=None, lrschedule='linear', save_dir = None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda : Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps
                                =nsteps, nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=
                                vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                                lrschedule=lrschedule)
    # Comments by Fei: change the directory to save models
#    if save_dir is not None:
#        with open(osp.join(save_dir, 'make_model.pkl'), 'wb') as fh:
#            pickle.dump(make_model, fh)
#    if save_interval and logger.get_dir():
#        import cloudpickle
#        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
#            fh.write(cloudpickle.dumps(make_model))
    
    model = make_model()
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)
    nbatch = nenvs*nsteps
    tstart = time.time()
    enqueue_threads = model.q_runner.create_threads(model.sess, coord=tf.train.Coordinator(), start=True)
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()

        # Comments by Fei: change the directory to save models
        if save_interval and (update % save_interval == 0 or update == 1) and save_dir:
            savepath = osp.join(save_dir, 'checkpoint%.5i'%update)
            print('Saving to ', savepath)
            model.save(savepath)
#        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
#            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i'%update)
#            print('Saving to', savepath)
#            model.save(savepath)

    env.close()