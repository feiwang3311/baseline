import os, time, pickle
import numpy as np
import scipy.sparse as sp
from minisat.minisat.gym.GymSolver import sat

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_PI(counts, tau):
    p = 1 / tau
    if p == 1.0:
        Pi = counts / np.sum(counts)
    elif p < 500:
        counts = counts / counts.sum()
        # do it in small steps to prevent underflow
        while p >= 10:
            counts = np.power(counts, 10)
            p = p / 10
            counts = counts / counts.sum()
        Pi = np.power(counts, p)
        Pi = Pi / np.sum(Pi)
    else:
       # assume that tau is infinitely small
       Pi = np.zeros(np.shape(counts), dtype = np.float32)
       Pi[np.argmax(counts)] = 1.0
    return Pi

class Pi_struct(object):
    """
        the object that holds a state and the PI array
    """
    def __init__(self):
    	self.state = None
    	self.Pi = None
        
    def add_state(self, state):
        """
            save the state as sparse matrix in 2D. Also compute the isValid array
        """
        self.isValid = np.reshape(np.any(state, axis = 0), [-1])
        assert np.any(self.isValid), "Error: state is empty!"
        state_2d = np.reshape(state, [-1, state.shape[1] * state.shape[2]])
        self.state = sp.csc_matrix(state_2d)

    def add_counts(self, counts, tau):
        """
            take the counts (of MCTS simulation from this state) and save PI. Also return the sampled next step
            counts: the counts of all the simulation steps
            tau:    the heat for computing Pi
        """
        assert counts.sum() == (counts * self.isValid).sum(), "count: " + str(counts) + \
        " is invalid: " + str(self.isValid) + " in file " + str(self.file_no)
        # tau is a function that takes the current level as parameter and returns the proper tau value for this level
        self.Pi = get_PI(counts, tau)

        assert (self.isValid * self.Pi).sum() > 0.999999, "Pi: " + str(self.Pi) + \
        " is invalid: " + str(self.isValid) + " in file " + str(self.file_no)
        # random select action based on Pi
        action = np.random.choice(len(self.Pi), 1, p = self.Pi)

        return action[0]

class Pi_structs(object):
    """
        this object maintains a list of Pi_struct for one self_play
    """
    def __init__(self, file_no, tau):
        """
            file_no: the file number of this SAT problem in the directory
            tau:     the function that returns a proper tau value given the current level
        """
        self.file_no = file_no
        self.tau = tau
        self.level = 0    # the current step in the self play, also the index into self.Pis
        self.Pis = []     # the list containing all the Pi_structs
        self.condition = "level points to a fresh spot"

    def add_state(self, state):	
        assert self.condition == "level points to a fresh spot", "Error: condition should be fresh"
        self.Pis.append(Pi_struct())
        self.Pis[self.level].add_state(state)
        self.condition = "level points to spot with state"

    def add_counts(self, counts):
        assert self.condition == "level points to spot with state", "Error: condition should be with state"
        action = self.Pis[self.level].add_counts(counts, self.tau(self.level))
        self.level += 1
        self.condition = "level points to a fresh spot"
        return action

    def dump_in_buffer(self, sl_Buffer):
        assert self.condition == "level points to a fresh spot", "Error: condition should be fresh"
        for pi in self.Pis:
            sl_Buffer.add(self.file_no, pi.state, pi.Pi, self.level)
        self.condition = "finish self play and saved to buffer"

class MCT(object):
    def __init__(self, file_path, file_no, max_clause1, max_var1, nrepeat, sl_buffer, tau, resign = 1000000):
        """
            file_path:   the directory to files that are used for training
            file_no:     the file index that this object works on (each MCT only focus on one SAT problem)
            max_clause1: the max_clause that should be passed to the env object
            max_var1:    the max_var that should be passed to the env object
            nrepeat:     the number of repeats that we want to self_play with this SAT problem
            sl_buffer:   the reference to the supervised learning storage buffer
            tau:         the function that, given the current number of steps, return a proper tau value
            resign:      the steps to pre-terminate as if done (usually too many steps)
        """
        self.env = sat(file_path, max_clause = max_clause1, max_var = max_var1)
        self.file_no = file_no
        self.state = self.env.resetAt(file_no)
        # IMPORTANT: all reset call should use the resetAt(file_no) function to make sure that it resets at the same file
        if self.state is None:
            self.phase = None # all nrepeats of this SAT problem are solved by simplification
        else:
            self.nrepeats = nrepeat
            self.tau = tau
            self.buffer = sl_buffer
            self.resign = resign
            self.working_repeat = 0 # working on the first repeat of this problem
            self.pi_structs = Pi_structs(self.file_no, self.tau)
            self.pi_structs.add_state(self.state)
            self.phase = False
            # phase False is "initial and normal running, should return state to neural nets"
            # phase True  is "pause and return state, should use results from neural nets for env simulation"
            # phase None  is "done and finished, should just return None"
        self.performance_list = []

    def get_state(self, pi_array, v_value):
        """
            main logic function:
            pi_array: the pi array evaluated by neural net (when phase is False, this paramete is not used)
            v_value:  the v  value evaluated by neural net (when phase is False, this paramete is not used)
            Return a state (3d numpy array) if paused for evaluation.
            Return None if this problem is simulated nrepeat times (all required repeat times are finished)
        """ 
        if self.phase is None: return None

        # loop for the simulation     
        needEnv = True
        while needEnv or needSim:
            if needEnv:
                if not self.phase:
                    self.phase = True
                    return self.state
                else:
                    self.phase = False
            self.state, needEnv, needSim = self.env.simulate(softmax(pi_array), v_value)

        # after simulation, save counts and make a step
        next_act = self.pi_structs.add_counts(self.env.get_visit_count())
        isDone, self.state = self.env.step(next_act)
        assert isDone or np.any(self.state), "isDone or state is not empty"
        if isDone or self.pi_structs.level >= self.resign:
            if self.buffer is not None:
                self.pi_structs.dump_in_buffer(self.buffer)       # save self play in buffer
            self.performance_list.append(self.pi_structs.level)   # save n_steps in the performance list
            self.working_repeat += 1                              # check how many repeats has been performed
            if self.working_repeat >= self.nrepeats:                  # if enough, set None and return None (mark finished)
                self.phase = None
                return None
            self.pi_structs = Pi_structs(self.file_no, self.tau)  # reset pi_structs
            self.state = self.env.resetAt(self.file_no)           # reset env and state
        # add to pi_structs the new state (could be after reset, could be normal step-foward)
        self.pi_structs.add_state(self.state)
        # route back to the start of the function
        return self.get_state(pi_array, v_value)

    def report_performance(self):
        if len(self.performance_list) == 0:
            return self.file_no, 1, 1
        return self.file_no, len(self.performance_list), sum(self.performance_list)