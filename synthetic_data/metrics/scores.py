"""
Compute accuracy
"""
import os
import psutil
import numpy as np
import pandas as pd
import pickle as pkl
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
	dist_file: string, optional
		The file that containts previously computed distances to omit recalculation.
	workers: int, optional
		The count of workers to use with the default value of 1.
	"""
	def __init__(self, train_file, test_file, synthetic_files, dist_file = None, workers = 1):
		"""
		Collect all training, testing and synthetic data files for processing
		"""

		training_data = pd.read_csv(train_file)
		training_data = training_data.fillna(training_data.mean())

		testing_data = pd.read_csv(test_file)
		testing_data = testing_data.fillna(testing_data.mean())

		self.data = {
						"training_data": training_data, 
					 	"testing_data": testing_data
				 	}

		self.synth_keys = []
		for i, s in enumerate(synthetic_files):
			self.data[f'synth_{i}'] = np.clip(pd.read_csv(s), 0, 1)
			self.synth_keys.append(f'synth_{i}')

		self.distances = {}

		if dist_file is not None:
			self.distances = pkl.load(open(dist_file, 'rb'))
		else:
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
		
		if not os.path.isdir("gen_data"):
			os.mkdir("gen_data")

		pkl.dump(self.distances, open(f'gen_data/syn_dists.pkl', 'wb'))

	def __discrepancy_score(self, t, s):
		left = np.mean(self.distances[(t, s)])
		right = np.mean(self.distances[(s, t)])
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

		print("Discrepency in training and test data is: {}".format(np.round(j_rr, 2)))
		print("Discrepency in training data and synthetic data is: {}".format(np.round(j_ra, 2)))
		print("Discrepency in testing and synthetic data is: {}".format(np.round(j_rat, 2)))
		print("Discrepency amongst various synthetic data files is: {}".format(np.round(j_aa, 2)))

	def __divergence(self, t, s):
		left = np.mean(np.log(self.distances[(t, s)] / self.distances[(t, t)]))
		right = np.mean(np.log(self.distances[(s, t)] / self.distances[(s, s)]))
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

		print("Divergence in training and synthetic data is: {}".format(np.round(training, 2)))
		print("Divergence in testing and synthetic data is: {}".format(np.round(testing, 2)))


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
			train_accuracy.append(self.__adversarial_accuracy('training_data', key))
			test_accuracy.append(self.__adversarial_accuracy('testing_data', key))

		avg_train_accuracy = np.mean(np.array(train_accuracy))
		avg_test_accuracy = np.mean(np.array(test_accuracy))
		return avg_train_accuracy, avg_test_accuracy

	def calculate_accuracy(self):
		"""
		Compute the standarad adversarial accuracy scores

		Outputs
		-------
		The adversarial accuracy for the two data files along with privacy loss.
		"""
		
		train_acc, test_acc = self.__calculate_accuracy()
		print("Adversarial accuracy for train data is: {}".format(np.round(train_acc, 2)))
		print("Adversarial accuracy for test data is: {}".format(np.round(test_acc, 2)))
		print("Privacy Loss is: {}".format(np.round(np.round(test_acc, 2) - np.round(train_acc, 2), 2)))
