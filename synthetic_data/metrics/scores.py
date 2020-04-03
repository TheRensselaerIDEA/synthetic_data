"""
Compute accuracy
"""
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from itertools import product
from sklearn.neighbors import NearestNeighbors

class Scores():
	""" 
	Calculates various scoring metrics amongst training, testing and synthetic data files.

	Parameters
	----------
	train_file : string, required
		The training file to be used.
	test_file : string, required
		The test file to be used.
	synthetic_file: list, required
		The list of various synthetic data files to be used.
	workers: int, optional
		The count of workers to use.
	"""
	def __init__(self, train_file, test_file, synthetic_files, workers=15):
		training_data = pd.read_csv(train_file)
		testing_data = pd.read_csv(test_file)

		self.data = {"training_data": training_data, 
					 "testing_data": testing_data}

		for i, s in enumerate(synthetic_files):
			self.data[f'synth_{i}'] = s
		self.synth_keys = [f'synth_{i}' for i in range(len(synthetic_files))]

		self.distances = {}

		self.__compute_nn(workers)

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

	def __discrepancy_score(self, t, s):
		left = np.mean(self.dists[(t, s)])
		right = np.mean(self.dists[(s, t)])
		return 0.5 * (left + right)

	def compute_discrepancy(self):
		"""
		Compute the standard discrepancy scores

		Outputs
		-------
		The discrepency amongst the various data files.
		"""
		j_rr = self.__discrepancy_score('training_data', 'testing_data')
		j_ra = []
		j_rat = []
		j_aa = []

		# For all of the synthetic datasets
		for k in self.synth_keys:
			j_ra.append(self.__discrepancy_score('training_data', k))
			j_rat.append(self.__discrepancy_score('testing_data', k))
			# Comparison to other synthetics
			for k_2 in self.synth_keys:
				if k != k_2:
					j_aa.append(self.__discrepancy_score(k, k_2))

		# Average across synthetics
		j_ra = np.mean(np.array(j_ra))
		j_rat = np.mean(np.array(j_rat))
		j_aa = np.mean(np.array(j_aa))

		print("Discrepency in training and test data is: {}".format(j_rr))
		print("Discrepency in training data and synthetic data is: {}".format(j_ra))
		print("Discrepency in testing and synthetic data is: {}".format(j_rat))
		print("Discrepency amongst various synthetic data files is: {}".format(j_aa))

	def __divergence(self, t, s):
		left = np.mean(np.log(self.dists[(t, s)] / self.dists[(t, t)]))
		right = np.mean(np.log(self.dists[(s, t)] / self.dists[(s, s)]))
		return 0.5 * (left + right)

	def compute_divergence(self):
		"""
		Compute the divergence scores

		Outputs
		-------
		The divergence score amongst the various data files.
		"""
		d_tr_a = []
		d_te_a = []

		for k in self.synth_keys:
			d_tr_a.append(self.__divergence('training_data', k))
			d_te_a.append(self.__divergence('testing_data', k))

		training = np.mean(np.array(d_tr_a))
		testing = np.mean(np.array(d_te_a))

		print("Divergence in training and synthetic data is: {}".format(training))
		print("Divergence in testing and synthetic data is: {}".format(testing))
