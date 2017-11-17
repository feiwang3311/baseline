import numpy as np
import pickle
import random
import scipy.sparse as sp
import itertools

filename = "SLRaw/saveSL"
nrepeat = 20
nfile = 3

data = []
with open(filename, "rb") as f:
	while True:
		try:
			data.append(pickle.load(f))
		except EOFError:
			break
print(len(data))

data2 = []
for i in range(len(data)):
	if i > 0 and data[i][2] == 0:
		data2.append(3)
	data2.append(data[i])
print(len(data2))


def splitby(iterable, indicator):
	return [list(g) for k, g in itertools.groupby(iterable, indicator) if not k]

data3 = splitby(data2, lambda x: isinstance(x, int))
print(len(data3))

for i in range(nfile):
	dataIn = zip(range(len(data3)), data3)
	dataF0 = list(filter(lambda x: x[0] % nfile == 0, dataIn))
	dataF0 = list(map(lambda x: x[1], dataF0))
	print(len(dataF0)) # nrepeat
	score = list(map(len, dataF0))
	print(score)
	# TODO: assign values based on the difference in score
	adjust_score = [0 for _ in range(len(score))]
	for j in range(len(dataF0)):
		for step in range(len(dataF0[j])):
			dataF0[j][step][2] = adjust_score[j]
	print(dataF0[0])
	exit(0)


class ReplayBufferSp(object):
	def __init__(self, size):
		"""Create Replay buffer that use sparse representation for saving memories.

		Parameters
		----------
		size: int
			Max number of transitions to store in the buffer. When the buffer
			overflows the old memories are dropped.
		"""
		self._storage = []
		self._maxsize = size
		self._next_idx = 0

	def __len__(self):
		return len(self._storage)

	def add(self, obs_t, policy, score):
		# Comments by Fei: obs_t should be [sp.csc_matrix, sp.csc_matrix], policy should be [probality], score should be scalar
		assert len(obs_t) == 2, "observations is not a list of len 2"
		data = (obs_t, policy, score)
		if self._next_idx >= len(self._storage):
			self._storage.append(data)
		else:
			self._storage[self._next_idx] = data
		self._next_idx = (self._next_idx + 1) % self._maxsize

	def _encode_sample(self, idxes):
		obses_t, policies, scores = [], [], []
		for i in idxes:
			data = self._storage[i]
			obs_t, policy, score = data
			# Comments by Fei: effort to convert observation (2 2D sparse) back to 3D numpy
			layer0 = obs_t[0].toarray()[:,:,None]
			layer1 = obs_t[1].toarray()[:,:,None]
			obs_t_trans = np.concatenate((layer0, layer1), axis = 2)

            obses_t.append(np.array(obs_t_trans, copy=False))
            policies.append(np.array(policy, copy=False))
            scores.append(score)
        return np.array(obses_t), np.array(policies), np.array(scores)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
