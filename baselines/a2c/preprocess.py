import pickle
import numpy as np
from random import shuffle

class cnf_snaps_reader(object):

	"""
		this is the initializer
		input_filenameList is a list of filenames that this object need to 
		read from
		output_filename is a filename that this object saves data to
		X_shape is a tuple (max_clause, max_var)
	"""
	def __init__(self, input_filenameList, output_filename, X_shape = None):
		self.input_filenameList = input_filenameList 
		self.output_filename = output_filename
		self.X_shape = X_shape
		if (X_shape != None):
			self.max_var = X_shape[1]
			self.max_clause = X_shape[0]
		else:
			self.max_var = -1 # mark as not initiated
			self.max_clause = -1 # mark as not initiated
		self.data_x_all = []
		self.data_y_all = []
		self.readAll()
		self.dumpAll()

	"""
		this function goes through all input files, and find the 
		max_var and max_clause, which are used for X_shape 
	"""
	def getXshape(self):
		for filename in self.input_filenameList:
			with open(filename, "r") as file:
				for line in file:
					if line.startswith("p cnf"):
						# this is the header of a cnf problem
						# get num_var and num_clause
						# p cnf 20 90
						header = line.split(" ")
						num_var = int(header[2])
						num_clause = int(header[3])
						if (self.max_var < num_var):
							self.max_var = num_var
						if (self.max_clause < num_clause):
							self.max_clause = num_clause
		self.X_shape = [self.max_clause, self.max_var]

	"""
		this function reads one file and put formatted data in fields
	"""
	def readFromOneFile(self, filename):
		if self.max_clause < 0:
			# X_shape is still not initialized, need to call getXshape
			self.getXshape()

		# read line from file and parse it to data format
		data_x = np.zeros((self.max_clause, self.max_var), dtype = np.int8)
		clause_counter = 0
		with open(filename, "r") as file:
			for l in file:
				if l.startswith("p cnf"):
					# this is the header of a cnf problem
					# p cnf 20 90
					header = l.split(" ")
					num_var = int(header[2])
					num_clause = int(header[3])
					if (num_clause == 0 or num_var == 0):
						# sometimes we get degenerated data at the end of the file
						break # should break
				    # initialize data_x and clause_counter 
					data_x = np.zeros((self.max_clause, self.max_var), dtype = int)
					clause_counter = 0
					assert (num_var <= self.max_var)
					assert (num_clause <= self.max_clause)

				elif l.startswith("c pick"):
					# this is the decision line, also the end of this data_x
					# c pick -1
					footer = l.split(" ")
					data_y = int(footer[2])
					self.data_x_all.append(data_x)
					self.data_y_all.append(data_y)

				elif l.startswith("c"):
				    # other comments line, skip
				    continue
				else:
					# clause data line
					# -11 -17 20 0
					literals = l.split(" ")
					n = len(literals)
					for j in range(n-1):
						value = 1 if int(literals[j]) > 0 else -1
						data_x[clause_counter, abs(int(literals[j])) - 1] = value
					clause_counter += 1
	"""
		this function reads all files and put data in fields
	"""
	def readAll(self):
		for filename in self.input_filenameList:
			self.readFromOneFile(filename)

	"""
		this function dumps the fields of formatted data by pickle
	"""
	def dumpAll(self):
		with open(self.output_filename, "wb") as dump:
			pickle.dump([np.asarray(self.data_x_all), np.asarray(self.data_y_all)], dump)

	"""
		this function pull from dump and show the first data point
	"""
	def verifyFirst(self):
		with open(self.output_filename, "rb") as pull:
			[X, Y] = pickle.load(pull)
			print (X.shape)
			print(Y.shape)
			print(np.linalg.norm(X[0]))
			print(np.linalg.norm(X[0][1]))
			print(X[0])
			print(Y[0])

class data_shuffler(object):
	"""
		this object shuffles the data in cnf_snaps_reader, and divide them into training and testing
		also reshape dataX into a vector for each data, and dataY into oneHot
	"""
	def __init__(self, input_dumpName, sortM = False):
		with open(input_dumpName, "rb") as pull:
			[dataX, dataY] = pickle.load(pull)
		self.num_data = dataY.shape[0]
		self.max_clause = dataX.shape[1]
		self.max_var = dataX.shape[2]
		
		# shuffle data
		index = [i for i in range(self.num_data)]
		shuffle(index)
		dataX_shuffle = dataX[index] # I think this is data copy
		dataY_shuffle = dataY[index] # I think this is data copy

		# if sortMatrix is true, sort them
		if (sortM):
			print("sort matrix")
			for i in range(self.num_data):
				dataX_shuffle[i] = self.sortMatrix(dataX_shuffle[i])

		# reshape data of each X as a vector
		dataX_reshape = np.reshape(dataX_shuffle, [self.num_data, -1])

		# make dataY one hot 
		dataY_onehot = np.zeros((self.num_data, 2 * self.max_var), dtype = int)
		for i in range(self.num_data):
			# old implementation, ignoring the sign of decision: dataY_onehot[i, abs(int(dataY_shuffle[i])) - 1] = 1
			value = int(dataY_shuffle[i])
			sign = 1 if value < 0 else 0
			dataY_onehot[i, 2 * abs(value) - 2 + sign] = 1

		# divide data into training and testing
		ratio = 0.8
		self.num_train = int(self.num_data * ratio)
		dataX_train = dataX_reshape[:self.num_train, :]
		dataY_train = dataY_onehot[:self.num_train, :]
		dataX_test = dataX_reshape[self.num_train:, :]
		dataY_test = dataY_onehot[self.num_train:, :]
		self.train = {"trainX": dataX_train, "trainY": dataY_train}
		self.test = {"testX": dataX_test, "testY": dataY_test}
		self.lastUsed = 0

	"""
		this function return a batch of trainig data
	"""	
	def next_batch(self, size):
		if self.lastUsed + size >= self.num_train:
			self.lastUsed = 0
		x = self.train["trainX"][self.lastUsed : self.lastUsed + size, :]
		y = self.train["trainY"][self.lastUsed : self.lastUsed + size, :]
		self.lastUsed += size
		return [x, y] 

	"""
		this function return the sorted Matrix 
	"""
	def sortMatrix(self, M):
		[row, col] = M.shape
		Morder = np.zeros(row)
		for i in range(col):
			Morder = Morder * 2 + np.absolute(M[:, i])
		index = np.argsort(-1 * Morder)
		return M[index, :]

def test_cnf_reader(fileOut):
	# get file list
	fileNameList = []
	with open("filelist", "r") as filelist:
		for line in filelist:
			fileNameList.append(line[:-1])
	# cnf read all files
	cnf = cnf_snaps_reader(fileNameList, fileOut, [100, 20])
	cnf.verifyFirst()

def test_shuffler():
	# cnf shuffle and reorder the data
	data = data_shuffler("tempOut")
	print(data.train)
	print(data.test)
	print(data.train["trainX"].shape)
	print(data.train["trainY"].shape)
	print(data.test["testX"].shape)
	print(data.test["testY"].shape)

def main():
	test_cnf_reader("training")


if __name__ == '__main__':
	main()
