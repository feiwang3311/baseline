import numpy as np
import random
import scipy.sparse as sp

class slBuffer(object):
    def __init__(self, size):
        """Create sl_buffer that use sparse representation for saving memories.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._nrepeat = []
        self._maxsize = size
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
