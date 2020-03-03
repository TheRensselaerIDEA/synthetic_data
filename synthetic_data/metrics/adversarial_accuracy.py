"""
Compute the nearest neighbor adversarial accuracy
"""

import os
import sys
from itertools import product
import pickle as pkl
import concurrent.futures
import psutil
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

class NearestNeighborMetrics():
    """ 
    Calculate nearest neighbors and metrics
    """

    def __init__(self, train_file, test_file, synthetic_file, prefix_synth):
        """ 
        The function initializes the Nearest Neighbors Metrics. 
  
        Parameters
        ----------
        train_file : string, required
            The training file to be used.
        test_file : string, required
            The test file to be used.
        synthetic_file: string, required
            The file that has the synthetic data.
        prefix_synth: string, required
            The name based on which everything should be saved.
        """

        p = psutil.Process()
        p.cpu_affinity(list(range(33, 48)))

        # Read in training, testing, and synthetic
        train = 
        test = 
        synthetics = []
        files = [
            f for f in os.listdir('.') if f.startswith(prefix_synth)
            and f.endswith('.csv') and 'synthetic' in f
        ]
        for f in files:
            synthetics.append(np.clip(pd.read_csv(synthetic_file), 0, 1))

        self.data = {'tr': pd.read_csv(train_file), 
                     'te': pd.read_csv(test_file),
                     'synth_0': pd.read_csv(synthetic_file)}
        self.synth_keys = ['synth_0']

        # Pre allocate distances
        self.dists = {}

        if f'{prefix_synth}_dists.pkl' not in os.listdir('.'):
            # Run all the calculations
            self.__compute_nn()

        # Run discrepancy score, divergence, adversarial accuracy
        discrepancy = self.__compute_discrepancy()
        divergence = self.__compute_divergence()
        adversarial = self.__compute_adversarial_accuracy()

        # Save to pickle
        pkl.dump({
            'discrepancy': discrepancy,
            'divergence': divergence,
            'adversarial': adversarial
        }, open(f'{prefix_synth}_results.pkl', 'wb'))

        pkl.dump(dists, open(f'{prefix_synth}_dists.pkl', 'wb'))

    def __nearest_neighbors(self, t, s):
        """
        Find nearest neighbors d_ts and d_ss
        """

        # Fit to S
        nn_s = NearestNeighbors(1).fit(self.data[s])
        if t == s:
            # Find distances from s to s
            d = nn_s.kneighbors()[0]
        else:
            # Find distances from t to s
            d = nn_s.kneighbors(self.data[t])[0]
        return t, s, d

    def __compute_nn(self):
        """
        Run all the nearest neighbors calculations
        """

        tasks = product(self.data.keys(), repeat=2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(self.__nearest_neighbors, t, s)
                for (t, s) in tasks
            ]
            # Wait for each job to finish
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures)):
                t, s, d = future.result()
                self.dists[(t, s)] = d

    def __divergence(self, t, s):
        """
        Calculate the NN divergence
        """
        left = np.mean(np.log(self.dists[(t, s)] / self.dists[(t, t)]))
        right = np.mean(np.log(self.dists[(s, t)] / self.dists[(s, s)]))

        return 0.5 * (left + right)

    def __discrepancy_score(self, t, s):
        """
        Calculate the NN discrepancy score
        """
        left = np.mean(self.dists[(t, s)])
        right = np.mean(self.dists[(s, t)])

        return 0.5 * (left + right)

    def __adversarial_accuracy(self, t, s):
        """
        Calculate the NN adversarial accuracy
        """
        left = np.mean(self.dists[(t, s)] > self.dists[(t, t)])
        right = np.mean(self.dists[(s, t)] > self.dists[(s, s)])

        return 0.5 * (left + right)

    def __compute_discrepancy(self):
        """
        Compute the standard discrepancy scores
        """
        j_rr = self.__discrepancy_score('tr', 'te')
        j_ra = []
        j_rat = []
        j_aa = []

        # For all of the synthetic datasets
        for k in self.synth_keys:
            j_ra.append(self.__discrepancy_score('tr', k))
            j_rat.append(self.__discrepancy_score('te', k))
            # Comparison to other synthetics
            for k_2 in self.synth_keys:
                if k != k_2:
                    j_aa.append(self.__discrepancy_score(k, k_2))

        # Average across synthetics
        j_ra = np.mean(np.array(j_ra))
        j_rat = np.mean(np.array(j_rat))
        j_aa = np.mean(np.array(j_aa))

        return j_rr, j_ra, j_rat, j_aa

    def __compute_divergence(self):
        """
        Compute the standard divergence scores
        """
        d_tr_a = []
        d_te_a = []

        for k in self.synth_keys:
            d_tr_a.append(self.__divergence('tr', k))
            d_te_a.append(self.__divergence('te', k))

        training = np.mean(np.array(d_tr_a))
        testing = np.mean(np.array(d_te_a))

        return training, testing

    def __compute_adversarial_accuracy(self):
        """
        Compute the standarad adversarial accuracy scores
        """
        a_tr_a = []
        a_te_a = []

        for k in self.synth_keys:
            a_tr_a.append(self.__adversarial_accuracy('tr', k))
            a_te_a.append(self.__adversarial_accuracy('te', k))

        a_tr = np.mean(np.array(a_tr_a))
        a_te = np.mean(np.array(a_te_a))

        return a_tr, a_te