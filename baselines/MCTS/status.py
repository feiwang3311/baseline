import pickle
import numpy as np

class Status(object):
	def __init__(self):
		"""
			create a Status object that keeps track of the status of MCTS RL algorithm
			NOTE: any changes to this Status object is updated in hardware by pickle dump
			NOTE: a new instance of Status can be created by reading from a pickle dump
		"""
		self.best_model = -1     # the index of the best model (if -1, means no model available)
		self.n_start = 0         # the index of the training file to start with for self_play
		self.ev_hist = []        # the list of evaluation history for all models
		self.length_hist = 0     # the length of models generated (or to evaluate)
		# NOTE: if length_hist is larger than len(ev_hist), it means some number of models have not been evaluated
		self.status_file = None  # the file name of this object

	def self_check(self):
		if self.best_model == -1: 
			assert self.length_hist == 0 and len(self.ev_hist) == 0, "when self.best_model is -1, self.length_hist should be 0, and self.ev_hist should be empty"
		assert self.best_model <= len(self.ev_hist), "self.best_model should be less than or equal to len(self.ev_hist)"
		assert len(self.ev_hist) <= self.length_hist, "self.ev_hist should have length less than or equal to self.length_hist"

	def write_to_disc(self):
		"""
			this method write the model to disc at self.status_file (every change should be updated) 
			TODO : optimize so that the vast content of ev_hist is not unnecessarily updated
		"""
		with open(self.status_file, "wb") as f:
			pickle.dump((self.best_model, self.n_start, self.ev_hist, self.length_hist, self.status_file), f)

	def start_with(self, status_file):
		"""
			this method fill the fields of Status object with information stored in a status pickle dump
		"""
		with open(status_file, "rb") as f:
			self.best_model, self.n_start, self.ev_hist, self.length_hist, self.status_file = pickle.load(f)
		self_check()
	
	def init_with(self, best_model, n_start, ev_hist, length_hist, status_file):
		"""
			this method initialize the fields of Status object with given parameters 
		"""
		self.best_model = best_model
		self.n_start = n_start
		self.ev_hist = ev_hist
		self.length_hist = length_hist
		self.status_file = status_file
		self_check()
		write_to_disc()

	def get_model_dir(self):
		"""
			this method returns the dir of the best model as indicated by self.best_model (or None of best_model == -1)
		"""
		if self.best_model == -1:
			return None
		return "model-" + str(self.best_model)

	def get_nbatch_index(self, nbatch, ntotal):
		"""
			this method update n_start field, with the help of nbatch and ntotal, and return the correct indexes of batch
		"""
		indexes = np.asarray(range(self.n_start, self.n_start + nbatch)) % ntotal
		self.n_start += nbatch
		self.n_start %= ntotal
		write_to_disc()
		return indexes

	def get_sl_starter(self):
		"""
			this method returns the starting model of the supervised learning
		"""
		assert self.length_hist > 0, "at supervised training stage, there should exist at least one model"
		return "model-" + str(self.length_hist - 1)

	def generate_new_model(self):
		"""
			this function is used by sl_train (or the initial phase of self_play) to put more models in hard drive
			it returns the new model dir name for this new model
		"""
		self.length_hist += 1
		write_to_disc()
		return "model-" + str(self.length_hist - 1)

	def which_model_to_evaluate(self):
		"""
			this function returns the dir name of the model to be evaluated
			it returns None if no such model dir exists
		"""
		index = len(self.ev_hist)
		if index < self.length_hist:
			return "model-" + str(index)
		else:
			return None

	def write_performance(self, performance):
		"""
			this function report the performance of the last evaluated model, then compare with the best model and possibally update it
		"""
		self.ev_hist.append(performance)
		if self.best_model == -1:
			assert len(self.ev_hist) == 1, "this must be the first model evaluated"
			self.best_model = 0
		else:
			if better_than(performance, self.ev_hist[self.best_model]):
				self.best_model = len(self.ev_hist) - 1
		write_to_disc()

	def better_than(per1, per2):
		if (per1 <= per2).sum() >= per1.shape[0] * 0.95 and np.mean(per1) < np.mean(per2) * 0.99:
			return True
		if (per1 <= per2).sum() >= per1.shape[0] * 0.65 and np.mean(per1) < np.mean(per2) * 0.95:
			return True
		if (per1 <= per2).sum() >= per1.shape[0] * 0.50 and np.mean(per1) < np.mean(per2) * 0.90:
			return True
		return False