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

class AdversarialAccuracy():
	""" 
	Calculates the adversarial accuracy between two data files

    Parameters
    ----------
    file_1 : string, required
        The first file to be considered for calculating accuracy.
    file_2 : string, required
        The second file to be considered for calculating accuracy.
    workers: int, optional
        The count of workers to use.
	"""
	def __init__(self, file_1, file_2, workers=15):
		file_1_data = pd.read_csv(file_1)
		file_1_data = file_1_data.fillna(file_1_data.mean())

		file_2_data = pd.read_csv(file_2)
		file_2_data = file_2_data.fillna(file_2_data.mean())

		self.data = {"file_1": file_1_data, 
					 "file_2": file_2_data}
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

	def __adversarial_accuracy(self, t, s):
		left = np.mean(self.distances[(t, s)] > self.distances[(t, t)])
		right = np.mean(self.distances[(s, t)] > self.distances[(s, s)])
		return 0.5 * (left + right)

	def calculate_accuracy(self):
		"""
		Compute the standarad adversarial accuracy scores

		Outputs
		-------
		The adversarial accuracy for the two data files.
		"""
		print("Adversarial accuracy is: {}".
			format(self.__adversarial_accuracy("file_1", "file_2")))