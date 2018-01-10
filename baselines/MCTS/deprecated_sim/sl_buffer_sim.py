import numpy as np
import random
import scipy.sparse as sp

class slBuffer_oneFile(object):
    def __init__(self, size, fileNo):
        """Create sl_buffer for one file at fileNo, that use sparse representation for saving memories.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        fileNo: int
            the file number for this buffer
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.fileNo = fileNo
        # this needs to be updated when data is added or removed. It helps calculate the actual score for each state/Pi pair
        self.mean_step = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs, Pi, step):
        data = (obs, Pi, step)

        if self._next_idx >= len(self._storage): # adding new data at new space!
            current_count = len(self._storage)
            self._storage.append(data)
            self.mean_step *= current_count * 1.0 / (current_count + 1)
            self.mean_step += step * 1.0 / (current_count + 1)
        else: # adding new data at old space, while removing an old data!
            _, _, old_step = self._storage[self._next_idx]
            self._storage[self._next_idx] = data
            self.mean_step += (step - old_step) * 1.0 / self._maxsize

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _get_score(self, step):
    	return np.tanh((self.mean_step - step) * 3.0 / self.mean_step)

    def _encode_sample(self, idxes):
        obses, Pis, scores = [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, Pi, step = data
            # Comments by Fei: effort to convert observation (2D sparse) back to 3D numpy"
            obs_2d = obs.toarray()
            obs_3d = np.reshape(obs_2d, [-1, int(obs_2d.shape[1] / 2), 2])

            obses.append(obs_3d)
            Pis.append(Pi)
            # Comments by Fei: effort to transform step into score
            scores.append(self._get_score(step))

        return np.array(obses, dtype = np.float32), np.array(Pis, dtype = np.float32), np.array(scores, dtype = np.float32)

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
        Pi_batch: np.array
            batch of actions executed given obs_batch
        score_batch: np.array
            rewards received as results of executing act_batch
        """
        idxes = np.random.choice(len(self._storage), batch_size)
        return self._encode_sample(idxes)

class slBuffer_allFile(object):
    def __init__(self, size, filePath, n_files):
        """
            this is a list of slBuffer_oneFile, which targets all files in filePath
            size:    the total size of this list of buffer. Need to divide by the number of files to get the size for each slBuffer_oneFile
            n_files: number of files in the filePath
        """
        self.filePath = filePath
        self.n_files = n_files
        self.totalSize = size
        self.eachSize = size // n_files
        self.bufferList = []
        for i in range(n_files): 
            self.bufferList.append(slBuffer_oneFile(self.eachSize, i))
        self.sample_round = -1
        self.sample_list = np.zeros(self.n_files, dtype = np.bool)

    def add(self, fromWhichFile, obs, Pi, step):
        self.bufferList[fromWhichFile].add(obs, Pi, step)
        self.sample_list[fromWhichFile] = True

    def sample(self, batch_size):
        assert np.any(self.sample_list), "Error: sample from an empty sl buffer"
        self.sample_round += 1
        self.sample_round %= self.n_files
        while not self.sample_list[self.sample_round]:
        	self.sample_round += 1
        	self.sample_round %= self.n_files
        return self.bufferList[self.sample_round].sample(batch_size)

import sys, pickle
if __name__ == '__main__':
    filename = sys.argv[1]
    with open(filename, "rb") as f:
        buffer1 = pickle.load(f)
    X, Y, Z = buffer1.sample(6)
    for i in range(6):
        print(np.sum(X[i]))
        print(Y[i])
        print(Z[i])