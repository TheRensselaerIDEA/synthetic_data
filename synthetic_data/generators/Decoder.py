"""
Module converts file into/out of SDV format
"""

import argparse
import json
import numpy as np
import numpy.random as rnd
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

class Decode():

	def __init__(self):
		pass

    def __read_decoders(self, prefix, npy_file):
        """read the decoder files"""
        limits = json.load(open(f"{prefix}.limits"))
        try:
            min_max = json.load(open(f"{prefix}.min_max"))
        except FileNotFoundError:
            min_max = None
        try:
            cols = json.load(open(f"{prefix}.cols"))
        except FileNotFoundError:
            cols = None
        if npy_file.endswith(".csv"):
            npy = pd.read_csv(npy_file)
        elif npy_file.endswith(".npy"):
            npy = np.load(npy_file)
        else:
            npy = None

        return limits, min_max, cols, npy

    def __read_data(self, file_name, dtype=None):
        """read in the file"""
        data = None
        if file_name.endswith(".csv") and dtype is not None:
            data = pd.read_csv(file_name, dtype = dtype)
        elif file_name.endswith(".csv"):
            data = pd.read_csv(file_name)
        elif file_name.endswith(".npy"):
            data = pd.DataFrame(np.load(file_name))

        # check if file can be read
        if data is None:
            raise ValueError

        return data

	def __undo_categorical(self, col, lim):
	    """convert a categorical column to continuous"""

	    def cat_decode(x, limits):
	        """decoder for categorical data"""
	        for k, v in limits.items():
	            if x <= float(k):
	                return v

	    return col.apply(lambda x: cat_decode(x, lim))


	def __undo_numeric(self, col, min_col, max_col, discrete=None):
	    """normalize a numeric column"""
	    if discrete:
	        return (((max_col - min_col) * col) + min_col).round().astype("int")
	    return ((max_col - min_col) * col) + min_col

	def __decode(self, df_new, df_orig_cols, limits, min_max):
	    """decode the data from SDV format"""
	    df_new = pd.DataFrame(df_new, columns=df_orig_cols)
	    for c in df_new.columns:
	        if c in limits:
	            df_new[c] = __undo_categorical(df_new[c], limits[c])
	        else:
	            df_new[c] = __undo_numeric(df_new[c], *min_max[c])

	    return df_new

	def decode_file(self, data_file, npy_file):
		lims, mm, cols, npy_new = __read_decoders(data_file[:-4], npy_file)
        if not cols:
            # open and read the data file
            df_raw = __read_data(data_file)
            cols = df_raw.columns

        df_converted = __decode(np.clip(npy_new, 0, 1), cols, lims, mm)
        # save decoded
        df_converted.to_csv(args.data_file[:-4] + "_synthetic.csv", index=False)