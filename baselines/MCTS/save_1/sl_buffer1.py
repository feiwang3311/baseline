import numpy as np
import random
import scipy.sparse as sp

class slBuffer_oneFile(object):
    def __init__(self, size, fileNo):
        """Create sl_buffer that use sparse representation for saving memories.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._maxsize = size
        self.fileNo = fileNo
        # storage related fields
        self._storage = []
        self._nrepeat = []
        self._next_idx = 0
        self._prob = None

    def __len__(self):
        return len(self._storage)

    def add(self, obs, Pi, score, repeat):
        self._prob = None
        # obs is alread 2d sparse array
        data = (obs, Pi, score)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            self._nrepeat.append(repeat)
        else:
            self._storage[self._next_idx] = data
            self._nrepeat[self._next_idx] = repeat
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses, Pis, scores = [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, Pi, score = data
            # Comments by Fei: effort to convert observation (2D sparse) back to 3D numpy"
            obs_2d = obs.toarray()
            obs_3d = np.reshape(obs_2d, [-1, int(obs_2d.shape[1] / 2), 2]) 
            
            obses.append(obs_3d)
            Pis.append(Pi)
            scores.append(score)
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
        if self._prob is None:
            self._prob = np.array(self._nrepeat) / sum(self._nrepeat)
        idxes = np.random.choice(len(self._storage), batch_size, p = self._prob)
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
        self.sample_round = -1 # this is the round-robin index for sample from the list of slBuffer_oneFile
        self.sample_list = np.zeros(self.n_files, dtype = np.bool)

    def add(self, fromWhichFile, obs, Pi, step, repeat):
        self.bufferList[fromWhichFile].add(obs, Pi, step, repeat)
        self.sample_list[fromWhichFile] = True

    def sample(self, batch_size):
        assert np.any(self.sample_list), "Error: sample from an empty sl buffer"
        self.sample_round += 1
        self.sample_round %= self.n_files
        while not self.sample_list[self.sample_round]:
            self.sample_round += 1
            self.sample_round %= self.n_files
        return self.bufferList[self.sample_round].sample(batch_size)