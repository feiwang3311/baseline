import os, time, pickle
import numpy as np
import scipy.sparse as sp
from minisat.minisat.gym.GymSolver import sat
from sl_buffer import slBuffer

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_PI(counts, tau):
    Pi  = np.power(counts, 1/tau)
    Pi = Pi / np.sum(Pi)
    return Pi

def get_nrepeat_count(action, nact):
    # given a numpy array of action, return an numpy array of the size of nact, and contains the counts of each element
    action = np.sort(action)
    counts = np.zeros(nact, dtype = int)
    base = 0
    start = 0
    for i in range(action.shape[0]):
        if action[i] > base:
            counts[base] = i - start
            start = i
            base = action[i]
    counts[base] = action.shape[0] - start
    print(counts)
    return counts

def analyze_Pi_graph_dump(Pi_node, sl_Buffer, standard):
    if not Pi_node.evaluated: return # this is the finished node (no state, not evaluated)
    for act in Pi_node.children:
        analyze_Pi_graph_dump(Pi_node.children[act], sl_Buffer, standard)
    # save this node's infor TODO: should the score be more quatified?
    av = Pi_node.score / Pi_node.repeat 
    if av > standard[1]: score = - (av-standard[1]) / (standard[2]-standard[1])
    elif av < standard[1]: score = (standard[1]-av) / (standard[1]-standard[0])
    else: score = 0
    sl_Buffer.add(Pi_node.state, Pi_node.Pi, score, Pi_node.repeat)

# this function is a hack that is no longer needed
def fix_invalid(Pi, isValid, file_no):
    """
        Pi:      the strengthened probibality array for stepping
        isValid: the array marking valid steps
        this function modify and return Pi, to make sure that all non-zero Pi values are for valid steps
    """
    n = isValid.shape[0]
    v = isValid.sum()
    total_valid = (isValid * Pi).sum()
    if total_valid < 0.999999: # errous state, report info for debugging
        print("ERROR: total_valid is only {} at file_no {}".format(total_valid, file_no))
        # fix the state
        Pi = (Pi + (1.0 - total_valid) / v) * isValid
    return Pi

class Pi_struct(object):
    def __init__(self, size, repeat, level, file_no, parent = None):
        self.size = size
        self.repeat = repeat
        self.level = level
        self.file_no = file_no
        self.children = {}
        self.next = 0
        self.evaluated = False
        self.parent = parent
        self.score = 0
    def add_state(self, state):
        self.isValid = np.reshape(np.any(state, axis = 0), [self.size,])
        state_2d = np.reshape(state, [-1, state.shape[1] * state.shape[2]])
        self.state = sp.csc_matrix(state_2d)
    def add_counts(self, counts):
        assert counts.sum() == (counts * self.isValid).sum(), "count: " + str(counts) + \
        " is invalid: " + str(self.isValid) + " in file " + str(self.file_no)
        # TODO: optimize on tau, which is a hyper parameter
        self.Pi = get_PI(counts, 0.9) 
        assert (self.isValid * self.Pi).sum() > 0.999999, "Pi: " + str(self.Pi) + \
        " is invalid: " + str(self.isValid) + " in file " + str(self.file_no)
        # random select actions based on Pi
        action = np.random.choice(range(self.size), self.repeat, p = self.Pi) 
        self.nrepeats = get_nrepeat_count(action, self.size)
        assert self.repeat == (self.nrepeats * self.isValid).sum(), "nrepeats: " + str(self.nrepeats) + \
        " is invalid: " + str(self.isValid) + " in file " + str(self.file_no)
        # create children of this Pi_struct
        for i in range(self.size):
            if self.nrepeats[i] > 0.5:
                self.children[i] = Pi_struct(self.size, self.nrepeats[i], self.level + 1, self.file_no, parent = self)
        self.evaluated = True
    # Deprecated
    def add_Pi(self, Pi, counts):
        assert counts.sum() == (counts * self.isValid).sum(), "count: " + str(counts) + " is invalid: " + str(self.isValid)
        self.Pi = np.copy(Pi)
        # hack: sometimes Pi has non-zero values in invalid actions, which is a problem (possibly bug in simulation process)
        # Because it happens in-frequently, for now, I try to fix Pi by isValid (pushing non-zero invalid Pi value to other valid positions)
        # also print something in the standoutput so that I know it happened, and hope to collect more info for debugging
        self.Pi = fix_invalid(self.Pi, self.isValid, self.file_no)

        action = np.random.choice(range(self.size), self.repeat, p = Pi) # random select actions based on Pi, for nrepeat times
        self.nrepeats = get_nrepeat_count(action, self.size)
        # same hack should be implemented for nrepeats. OR at least an assert check here to make sure that all nrepeats non-zero are valid!!
        assert self.repeat == (self.nrepeats * self.isValid).sum(), "all non-zero nrepeats must be valid"
        for i in range(self.size):
            if self.nrepeats[i] > 0.5:
                self.children[i] = Pi_struct(self.size, self.nrepeats[i], self.level+1, self.file_no, parent = self)
        self.evaluated = True
    def get_next(self):
        while self.next < self.size and (self.nrepeats[self.next] < 0.5 or not self.isValid[self.next]):
            self.next += 1
        if self.next >= self.size: 
            return -1 # no more actions to go
        return self.next
    def set_next(self, additional_score):
        # print("node at level {} added score of worth {}".format(self.level, additional_score))
        self.score += additional_score
        self.next += 1
        next = self.get_next()
        if (self.get_next() < 0 and self.parent is not None):
            return self.parent.set_next(self.score)
        return next

class MCT(object):
    def __init__(self, file_path, file_no, max_clause1, max_var1, nrepeat):
        """
            file_path:   the directory to files that are used for training
            file_no:     the file index that this object works on (each MCT only focus on one file problem)
            max_clause1: the max_clause that should be passed to the env object
            max_var1:    the max_var that should be passed to the env object
            nrepeat:     the number of repeats that we want to self_play with this file problem (suggest 100)
        """
        self.env = sat(file_path, max_clause = max_clause1, max_var = max_var1) 
        self.Pi_current = self.Pi_root = Pi_struct(max_var1 * 2, nrepeat, 0, file_no)
        self.state = self.env.resetAt(file_no) ## all reset call should use the function with file_no as parameter, to make sure that it always uses the same file
        self.Pi_current.add_state(self.state)
        self.min_step = 10000000
        self.max_step = 0
        self.file_no = file_no
        self.phase = False # phase False is "initial and normal running" phase, phase True is "pause and return state" phase

    def get_state(self, pi_array, v_value):
        """
            main logic function:
            pi_array: the pi array evaluated by neural net (when phase is False, this paramete is not used)
            v_value:  the v value evaluated by neural net  (when phase is False, this paramete is not used)
            Return a state (3d numpy array) if paused for evaluation.
            Return None if this problem is simulated nrepeat times.
        """
        while True:
            if not self.Pi_current.evaluated:
                if not self.phase:
                    print("simulation -----------------------------------------------{} at level {}".format(self.Pi_current.repeat, self.Pi_current.level))
                needEnv = True
                while needEnv or needSim:
                    if needEnv:
                        if not self.phase:
                            self.phase = True
                            return self.state
                        else:
                            self.phase = False
                    self.state, needEnv, needSim = self.env.simulate(softmax(pi_array), v_value)
                self.Pi_current.add_counts(self.env.get_visit_count())

            next_act = self.Pi_current.get_next() 
            assert next_act >= 0, "next_act is neg in file " + str(self.file_no)
            isDone, self.state = self.env.step(next_act) # there is a guarantee that this next_act is valid
            if isDone:
                print("step {} to DONE+++++++++++++++++++++++++++++++++++++++++++++++++++++".format(next_act))
                if self.Pi_current.level < self.min_step: self.min_step = self.Pi_current.level # update score range for this sat prob
                if self.Pi_current.level > self.max_step: self.max_step = self.Pi_current.level
                if (self.Pi_current.set_next(self.Pi_current.level * self.Pi_current.nrepeats[next_act]) < 0): # write back the total score from this leaf node
                    return None # we are finished
                self.Pi_current = self.Pi_root
                self.state = self.env.resetAt(self.file_no)
            else:
                print("step {} is continuing***********************************************".format(next_act))
                self.Pi_current = self.Pi_current.children[next_act]
                self.Pi_current.add_state(self.state)

    def write_data_to_buffer(self, sl_Buffer):
        standard = (self.min_step, self.Pi_root.score / self.Pi_root.repeat, self.max_step)
        analyze_Pi_graph_dump(self.Pi_root, sl_Buffer, standard)