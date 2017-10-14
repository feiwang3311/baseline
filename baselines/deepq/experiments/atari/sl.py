import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time
import json
import pickle
from random import shuffle

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines.deepq.experiments.atari.model import model, dueling_model

def parse_args():
    parser = argparse.ArgumentParser("Supervised Learning for deepq neural network in SAT solving")
    # Environment
    parser.add_argument("--env", type=str, default="gym_sat_Env-v0", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Comments by Fei: about environment, are we in test mode with a test_path?
    parser.add_argument("--test_path", type=str, default=None, help="if in the test mode, give the directory of SAT problems for testing")
    parser.add_argument("--dump_pair_into", type=str, default="SLData", help="if in the test mode, give the directory of saving state-action pairs")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(2e8), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=64, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000, help="number of iterations between every target network update")
    parser.add_argument("--param-noise-update-freq", type=int, default=50, help="number of iterations between every re-scaling of the parameter noise")
    parser.add_argument("--param-noise-reset-freq", type=int, default=10000, help="maximum number of steps to take per episode before re-perturbing the exploration policy")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    boolean_flag(parser, "param-noise", default=False, help="whether or not to use parameter space noise for exploration")
    boolean_flag(parser, "layer-norm", default=False, help="whether or not to use layer norm (should be True if param_noise is used)")
    boolean_flag(parser, "gym-monitor", default=False, help="whether or not to use a OpenAI Gym monitor (results in slower training due to video recording)")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./SLsave_normal_train", help="directory in which training model should be saved.")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=1e4, help="save model once every time this many iterations are completed")
    parser.add_argument("--load-model", type=int, default=-1, help="if not negative, load model number load-model before supervised training")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()

class data_shuffler(object):
	"""
		this object shuffles the data in pickle_dump, and divide them into training and testing
	"""
	def __init__(self, input_dumpName, sortM = False, nbatch = -1):
		with open(input_dumpName, "rb") as pull:
			data_dict = pickle.load(pull)
		dataX = np.asarray(data_dict["states"]) # should be numpy array of ndata * max_clause * max_var * 1
		dataY = np.asarray(data_dict["actions"]) # should be numpy array of ndata, each content is int of choice (index)
		(self.num_data, self.max_clause, self.max_var, _) = dataX.shape
		
		# shuffle data
		index = [i for i in range(self.num_data)]
		shuffle(index)
		dataX_shuffle = dataX[index] # I think this is data copy
		dataY_shuffle = dataY[index] # I think this is data copy

		# divide data into training and testing (May want to optimize on training size)
		ratio = 0.8
		if nbatch > 0: # nbatch is given at construction time, smartly adjust num_train to be a multiple of nbatch
			self.num_train = int(self.num_data * ratio) // nbatch * nbatch 
		if nbatch == -1: # no nbatch information given
			self.num_train = int(self.num_data * ratio)
		dataX_train = dataX_shuffle[:self.num_train, :, :, :]
		dataY_train = dataY_shuffle[:self.num_train]
		dataX_test = dataX_shuffle[self.num_train:, :, :, :]
		dataY_test = dataY_shuffle[self.num_train:]
		self.train = {"trainX": dataX_train, "trainY": dataY_train}
		self.test = {"testX": dataX_test, "testY": dataY_test}
		self.lastUsed = 0

	"""
		this function return a batch of trainig data
	"""	
	def next_batch(self, size):
		if self.lastUsed + size >= self.num_train:
			self.lastUsed = 0
		x = self.train["trainX"][self.lastUsed : self.lastUsed + size, :, :, :]
		y = self.train["trainY"][self.lastUsed : self.lastUsed + size]
		self.lastUsed += size
		return [x, y] 

def maybe_save_model(savedir, model_num):
    """This function checkpoints the model of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(model_num)
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))

def maybe_load_model(savedir, model_num):
    """Load model if present at the specified path."""
    if savedir is None:
        return
    model_dir = "model-{}".format(model_num) # Comments by Fei: for whatever change in maybe_save_model, reflect it here!
    U.load_state(os.path.join(savedir, model_dir, "saved"))

def super_train(filename, num_steps, nbatch, num_report, layer_norm, num_procs, num_model, load_model_num = -1):
	"""
		this function trains a CNN by supervised learning
		filename: the name of file that pickle dumped the training and testing data ind_flat_filter
		num_steps: total number of steps of supervised training
		nbatch: number of samples used per training step 
		num_report: by how many steps should we output the training and testing accuracy
		layer_norm: boolean flag of whether we want to normalize each layer
		num_procs: number of processors used in this training
		num_model: how often should model name change
		load_model_num: if not negative, load load_model_num at args.save_dir before supervised training
	"""

	print("read files from pickle_dump file %s" % (filename))
	data = data_shuffler(filename, nbatch)
	num_var = int(data.max_var)
	num_clause = int(data.max_clause)
	num_actions = num_var * 2
	
	observations_ph = tf.placeholder(tf.float32, shape = [None, num_clause, num_var, 1], name = "states")
	y_ = tf.placeholder(tf.int64, shape = [None], name = "actions")

	# construct the model
	kwargs = {}
	with tf.variable_scope("deepq", reuse=None):
		# Comments by Fei: this is nbatch * nact, with values as q values
		q_values = model(observations_ph, num_actions, scope="q_func", layer_norm=layer_norm, **kwargs) 

		# filter out non-valid actions
		pos = tf.reduce_max(observations_ph, axis = 1) # Comments by Fei: get 1 if the postive variable exists in any clauses, otherwise 0
		neg = tf.reduce_min(observations_ph, axis = 1) # Comments by Fei: get -1 if the negative variables exists in any clauses, otherwise 0
		ind = tf.concat([pos, neg], axis = 2) # Comments by Fei: get (1, -1) if this var is present, (1, 0) if only as positive, (0, -1) if only as negative
		ind_flat = tf.reshape(ind, [-1, num_actions]) # Comments by Fei: this is nbatch * nact, with 0 values labeling non_valid actions, 1 or -1 for other
		ind_flat_filter = tf.abs(tf.cast(ind_flat, tf.float32)) # Comments by Fei: this is nbatch * nact, with 0 values labeling non_valid actions, 1 for other
		q_min = tf.reduce_min(q_values, axis = 1)
		q_values_adjust = q_values - tf.expand_dims(q_min, axis = 1) # Comments by Fei: make sure the maximal values are positive
		q_values_filter = q_values_adjust * ind_flat_filter # Comments by Fei: zero-fy non-valid values, unchange valid values

	# Train and evaluate
	with tf.name_scope("loss"):
		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_, logits = q_values_filter))
	with tf.name_scope("train"):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	with tf.name_scope("accuracy"):
		correct_prediction = tf.equal(tf.argmax(q_values_filter, 1), y_)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=num_procs, inter_op_parallelism_threads=num_procs)
	config.gpu_options.allow_growth = True

	# run training in session
	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())

		# may load model if given
		save_dir = args.save_dir
		if load_model_num >= 0: # there is a model to load before training
			print("load model number {} before supervised learning".format(load_model_num))
			maybe_load_model(save_dir, load_model_num)

		# supervised training cycle
		for i in range(num_steps):
			batch = data.next_batch(nbatch)
			feed_dict={observations_ph: batch[0], y_: batch[1]}
			sess.run(train_step, feed_dict)
			if i > 0 and i % num_report == 0: # report accuracy
				train_accuracy = sess.run(accuracy, feed_dict)
				print('step %d, training accuracy %g' % (i, train_accuracy))
				feed_dict_test = {observations_ph: data.test["testX"], y_: data.test["testY"]}
				test_accuracy = sess.run(accuracy, feed_dict_test)
				print('step %d, testing accuracy %g' % (i, test_accuracy))
				# save model
				maybe_save_model(save_dir, i // num_model + load_model_num if load_model_num >= 0 else 0)

if __name__ == '__main__':
	args = parse_args()
	filename = args.dump_pair_into
	num_steps = args.num_steps
	nbatch = args.batch_size
	num_report = args.save_freq
	layer_norm = args.layer_norm
	num_procs = 16
	num_model = 1e6
	load_model_num = args.load_model
	super_train(filename, num_steps, nbatch, num_report, layer_norm, num_procs, num_model, load_model_num = load_model_num)