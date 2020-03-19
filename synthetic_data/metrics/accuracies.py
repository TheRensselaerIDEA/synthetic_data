"""
Compute accuracy
"""
import psutil
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import concurrent.futures
from itertools import product
from sklearn.neighbors import NearestNeighbors

class AdversarialAccuracy():
	""" 
	Calculates the adversarial accuracy between two data files

	Parameters
	----------
	train_file : string, required
		The train file to be considered for calculating accuracy.
	test_file : string, required
		The test file to be considered for calculating accuracy.
	synth_files : list, required
		The list of synthetic files to be used for calculating accuracy.
	workers: int, optional
		The count of workers to use (default is 15).
	"""

	def __init__(self, train_file, test_file, synth_files, workers=15):
		"""
		Collect all training, testing and synthetic data files for processing
		"""

		train_data = pd.read_csv(train_file)
		train_data = train_data.fillna(train_data.mean())

		test_data = pd.read_csv(test_file)
		test_data = test_data.fillna(test_data.mean())

		self.data = {"train": train_data, "test": test_data}

		self.synth_keys = []
		for i, s in enumerate(synth_files):
			self.data[f'synth_{i}'] = np.clip(pd.read_csv(s), 0, 1)
			self.synth_keys.append(f'synth_{i}')

		self.distances = {}

		self.workers = workers

	def __nearest_neighbors(self, t, s):
		# Fit to S
		nn_s = NearestNeighbors(1).fit(self.data[s])
		if t == s:
			# Find distances from s to s
			d = nn_s.kneighbors()[0]
		else:
			# Find distances from t to s
			d = nn_s.kneighbors(self.data[t])[0]
		return t, s, d

	def __compute_nn(self, workers):
		tasks = product(self.data.keys(), repeat=2)

		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
			futures = [
				executor.submit(self.__nearest_neighbors, t, s)
				for (t, s) in tasks
			]

			# Wait for each job to finish
			for future in tqdm(concurrent.futures.as_completed(futures),
							   total=len(futures)):
				t, s, d = future.result()
				self.distances[(t, s)] = d

		pkl.dump(self.distances, open(f'gen_data/syn_dists.pkl', 'wb'))

	def __adversarial_accuracy(self, t, s):
		left = np.mean(self.distances[(t, s)] > self.distances[(t, t)])
		right = np.mean(self.distances[(s, t)] > self.distances[(s, s)])
		return 0.5 * (left + right)

	def __calculate_accuracy(self):
		"""
		Compute the standarad adversarial accuracy scores
		"""

		train_accuracy = []
		test_accuracy = []
		for key in self.synth_keys:
			train_accuracy.append(self.__adversarial_accuracy('train', key))
			test_accuracy.append(self.__adversarial_accuracy('test', key))

		avg_train_accuracy = np.mean(np.array(train_accuracy))
		avg_test_accuracy = np.mean(np.array(test_accuracy))
		return avg_train_accuracy, avg_test_accuracy

	def calculate_accuracy(self, dist_file=None):
		"""
		Compute the standarad adversarial accuracy scores

		Parameters
		----------
			dist_file : string, optional
				The file that containts previously computed distances 
				to omit recalculation.
		Outputs
		-------
			The adversarial accuracy for the two data files.
		"""

		if dist_file is not None:
        	self.distances = pkl.load(open(dist_file, 'rb'))
        else:
			self.__compute_nn(self.workers)
		
		train_acc, test_acc = self.__calculate_accuracy()
		print("Adversarial accuracy for train data is: {}".format(train_acc))
		print("Adversarial accuracy for test data is: {}".format(test_acc))