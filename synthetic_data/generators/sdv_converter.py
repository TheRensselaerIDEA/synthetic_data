"""
The module converts CSV/NPY files into/from SDV using
the encoder and decoder classes.
"""

import argparse
import json
import numpy as np
import numpy.random as rnd
import pandas as pd
from scipy import stats
from progress.bar import IncrementalBar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

class Encoder():
    """ 
    Encode train/test files.

    The class provides functions to convert a Comma Separated 
    Values (CSV) file or Numpy (NPY) file into SDV file as 
    described below:
    - SDV file
    - Limits file
    - Min_max file
    """
    def __init__(self):
        pass

    def __read_data(self, file_name, dtype=None):
        """
        Read in the file which can be CSV or Numpy
        """
        data = None
        if file_name.endswith(".csv") and dtype is not None:
            data = pd.read_csv(file_name, dtype = dtype)
        elif file_name.endswith(".csv"):
            data = pd.read_csv(file_name)
        elif file_name.endswith(".npy"):
            data = pd.DataFrame(np.load(file_name))

        # Check if file can be read
        if data is None:
            raise ValueError

        return data

    def __impute_column(self, df, c):
        """
        Impute a column (c) in dataframe using 
        LinearRegression if it is continuous or 
        KNeighborsClassifier is it is not
        """

        # Get X and y
        y = df[c]
        X = df.drop(c, axis=1)

        # Remove columns with in the values to impute
        X = X.loc[:, ~(X[y.isna()].isna().any())]

        # Remove rows with NA values
        na_mask = ~(X.isna().any(axis=1))
        y = y[na_mask]
        X = X[na_mask]

        # One hot encode the data
        X = one_hot_encode(X)

        # Get mask for data to impute
        impute_mask = y.isna()

        # If y is continuous then use linear regression
        # else use KNeighborsClassifier
        if y.dtype.name == "float64":
            clf = LinearRegression()
        elif y.dtype.name == "object":
            clf = KNeighborsClassifier(3, weights="distance")
        else:
            raise ValueError

        trained_model = clf.fit(X[~impute_mask], y[~impute_mask])
        imputed_values = trained_model.predict(X[impute_mask])

        return imputed_values

    def __fix_na_values(self, df, cols_ignored):
        """
        Impute missing values in the columns
        """
        df_core = df.drop(cols_ignored, axis=1)

        while df_core.isna().sum().sum():
            # Get column with least amount of missing values
            cols_with_na = df_core.isna().sum()
            col = cols_with_na[cols_with_na > 0].idxmin()
            # Impute that column
            df_core.loc[df_core[col].isna(), col] = self.__impute_column(df_core, col)

        return pd.concat([df_core, df[cols_ignored]], axis=1)

    def __truncated_beta(self, alpha, beta, low, high):
        """
        Perform truncated beta distribution 
        with params - alpha and beta
        and limits - low and high
        """
        nrm = stats.beta.cdf(high, alpha, beta) - stats.beta.cdf(low, alpha, beta)

        low_cdf = stats.beta.cdf(low, alpha, beta)

        while True:
            yr = rnd.random(1) * nrm + low_cdf
            xr = stats.beta.ppf(yr, alpha, beta)
            yield xr[0]

    def __binary(self, col, limits=None):
        """
        Convert a binary column to a continuous column
        """
        if limits:
            # Construct the distributions
            distributions = {}

            zeros = min(limits.keys())

            # Handling the case of all zeros or/and all ones
            if zeros == 1:
                return col.apply(lambda x: 0), {1.0: 0}
            if zeros == 0:
                return col.apply(lambda x: 1), {1.0: 1}

            alpha = (zeros) * 100
            beta = ((len(col) - (zeros * len(col))) / len(col)) * 100

            distributions[0] = self.__truncated_beta(alpha, beta, 0, zeros)

            distributions[1] = self.__truncated_beta(alpha, beta, zeros, 1)

            # Convert values that don't exist in original column
            # to most common value
            col = col.copy()

            return col.apply(lambda x: next(distributions[x])), None

        zeros = (col == 0).sum() / len(col)
        alpha = zeros * 100
        beta = ((len(col) - (col == 0).sum()) / len(col)) * 100

        # Handling the case of all zeros or/and all ones
        if zeros == 1:
            return col.apply(lambda x: 0), {1.0: 0}
        if zeros == 0:
            return col.apply(lambda x: 1), {1.0: 1}

        # Get distributions to pull from
        distributions = {}
        limits = {}

        distributions[0] = self.__truncated_beta(alpha, beta, 0, zeros)
        limits[zeros] = 0

        distributions[1] = self.__truncated_beta(alpha, beta, zeros, 1)
        limits[1] = 1

        # Sample from the distributions and return that value
        return col.apply(lambda x: next(distributions[x])), limits

    def __numeric(self, col, min_max=None):
        """
        Normalize a numeric column
        """
        if min_max:
            return ((col - min_max[0]) / (min_max[1] - min_max[0])), None, None
        return ((col - min(col)) / (max(col) - min(col))), min(col), max(col)

    def __categorical(self, col, limits=None):
        """
        Convert a categorical column to continuous column
        """
        if limits:
            # Construct the distributions
            distributions = {}
            a = 0
            for b, cat in limits.items():
                b = float(b)
                mu, sigma = (a + b) / 2, (b - a) / 6
                distributions[cat] = stats.truncnorm(
                    (a - mu) / sigma, (b - mu) / sigma, mu, sigma
                )
                a = b

            # Convert values that don't exist in original column
            # to most common value
            col = col.copy()
            common = col.value_counts().index[0]
            for cat in col.unique():
                if cat not in distributions:
                    col.loc[col == cat] = common

            return col.apply(lambda x: distributions[x].rvs()), None

        # Get categories (ensures sort by value and then name to tie-break)
        series = col.value_counts(normalize=True)
        tmp = pd.DataFrame({"names": series.index, "pcts": series.values})
        tmp = tmp.sort_values(["pcts", "names"], ascending=[False, True])
        categories = pd.Series(tmp.pcts.values, tmp.names.values)

        # Get distributions to pull from
        distributions = {}
        limits = {}
        a = 0

        # Iterate for each category
        for cat, val in categories.items():
            # Identify the cut off value
            b = a + val
            # Create the distribution to sample from
            mu, sigma = (a + b) / 2, (b - a) / 6
            distributions[cat] = stats.truncnorm(
                (a - mu) / sigma, (b - mu) / sigma, mu, sigma
            )
            limits[b] = cat
            a = b

        # Sample from the distributions and return that value
        return col.apply(lambda x: distributions[x].rvs()), limits

    def __ordinal(self, col, limits=None):
        """
        Convert an ordinal column to a continuous column
        """
        if limits:
            # Construct the distributions
            distributions = {}
            a = 0
            for b, cat in limits.items():
                b = float(b)
                mu, sigma = (a + b) / 2, (b - a) / 6
                distributions[cat] = stats.truncnorm(
                    (a - mu) / sigma, (b - mu) / sigma, mu, sigma
                )
                a = b

            # Convert values that don't exist in original column
            # to nearest value
            col = col.copy()
            max_val = max(distributions.keys())
            min_val = min(distributions.keys())
            for cat in col.unique():
                if cat not in distributions:
                    if cat > max_val:
                        col.loc[col == cat] = max_val
                    else:
                        col.loc[col == cat] = min_val

            return col.apply(lambda x: distributions[x].rvs()), None

        # Get categories, ensures sort by value and then name to tiebreak
        categories = col.value_counts()

        # Find missing categories and impute them with zeroes
        for i in range(categories.keys().min(), categories.keys().max() + 1):
            if i not in categories.keys():
                categories[i] = 0

        # Sort by index
        categories = categories.sort_index()

        # Additive smoothing for 0 counts
        alpha = 1
        new_vals = (categories.values + alpha) / (len(col) + (alpha * len(categories)))

        # Create new categories
        categories = pd.Series(new_vals, index=categories.index)

        # Get distributions to pull from
        distributions = {}
        limits = {}
        a = 0

        # Iterate for each category
        for cat, val in categories.items():
            # Figure out the cutoff value
            b = a + val
            # Create the distribution to sample from
            mu, sigma = (a + b) / 2, (b - a) / 6
            distributions[cat] = stats.truncnorm(
                (a - mu) / sigma, (b - mu) / sigma, mu, sigma
            )
            limits[b] = cat
            a = b

        # Sample from the distributions and return that value
        return col.apply(lambda x: distributions[x].rvs()), limits

    def __encode(self, df, limits=None, min_max=None, beta=False):
        """
        encode the data into SDV format
        """
        # Loop through every column
        if limits and min_max:
            already_exists = True
        else:
            limits = {}
            min_max = {}
            already_exists = False
        for c in df.columns:
            # If column is "object"
            if df[c].dtype.char == "O":
                if already_exists:
                    df[c], _ = self.__categorical(df[c], limits[c])
                else:
                    df[c], lim = self.__categorical(df[c])
                    limits[c] = lim
            # If column is "int"
            elif df[c].dtype.char == "l" or df[c].dtype.char == "q":
                # If column is "binary"
                if set(df[c].unique()).issubset(set((0, 1))):
                    if already_exists:
                        if beta:
                            df[c], _ = self.__binary(df[c], limits[c])
                        else:
                            df[c], _ = self.__categorical(df[c], limits[c])
                    else:
                        if beta:
                            df[c], lim = self.__binary(df[c])
                        else:
                            df[c], lim = self.__categorical(df[c])
                        limits[c] = lim
                # If column is "ordinal"
                else:
                    if already_exists:
                        df[c], _ = self.__ordinal(df[c], limits[c])
                    else:
                        df[c], lim = self.__ordinal(df[c])
                        limits[c] = lim
            # If column is "boolean"
            elif df[c].dtype.char == "?":
                if already_exists:
                    if beta:
                        df[c], _ = self.__binary(df[c], limits[c])
                    else:
                        df[c], _ = self.__categorical(df[c], limits[c])
                else:
                    if beta:
                        df[c], lim = self.__binary(df[c])
                    else:
                        df[c], lim = self.__categorical(df[c])
                    limits[c] = lim

            # If column is "decimal"
            elif df[c].dtype.char == "d":
                if already_exists:
                    df[c], _, _ = self.__numeric(df[c], min_max[c])
                else:
                    df[c], min_res, max_res = self.__numeric(df[c])
                    min_max[c] = (min_res, max_res, 0)

        return df, limits, min_max

    def __save_files(self, df, prefix, limits=None, min_max=None, cols=False):
        """
        Save the sdv file and decoders
        """
        df.to_csv(f"{prefix}_sdv.csv", index=False)
        if cols:
            json.dump(df.columns.tolist(), open(f"{prefix}.cols", "w"))
        if limits:
            json.dump(limits, open(f"{prefix}.limits", "w"))
        if min_max:
            json.dump(min_max, open(f"{prefix}.min_max", "w"))

    def __read_decoders(self, prefix, npy_file):
        """
        Read the decoder files
        """
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

    def encode_train(self, 
                    data_file,
                    fix_na_values=False, 
                    na_col_to_ignore=[], 
                    dtype=None,
                    beta=False):

        """ 
        The function encodes the training file into SDV file. 
  
        Parameters
        ----------
        data_file : str, required
            The training file as CSV or NPY which needs to be converted.
        fix_na_values: boolean, optional
            Boolean variable which imputes the rows that have NA values in them (default is False.
        na_col_to_ignore: list, optional
            If the parameter "fix_na_values" is set to True, this list of columns will be ignored from imputation (default is empty list).
        dtype: dictionary, optional
            If you want to specify which column should be treated as continuous and which one as numeric, you can use Pandas' dtype dictioary for each column (default is None).
        beta: boolean, optional
            Use binary imputation rather than categorical for categorical columns (default is False).
          
        Outputs
        -------
        SDV file:
            The converted sdv file of the original training file provided with original name appended with "_sdv".
        Limits file:
            The limits file with the name same as original file but with extension as "limits".
        Min-max file:
            The min-max file with the name same as original file but with extension as "min_max".
        """

        bar = IncrementalBar('Encoding', max=8)
        # Open and read the data file
        bar.next()
        df_raw = self.__read_data(data_file, dtype)
        bar.next()
        bar.next()
        bar.next()

        if fix_na_values:
            # Fix the NA values
            df_raw = self.__fix_na_values(df_raw, na_col_to_ignore)
            assert df_raw.isna().sum().sum() == 0

        bar.next()
        bar.next()
        bar.next()
        df_converted, lims, mm = self.__encode(df_raw, beta)
        bar.next()
        self.__save_files(df_converted, data_file[:-4], lims, mm, True)
        bar.finish()

    def encode_test(self, 
                    data_file,
                    encoder_file,
                    fix_na_values=False,
                    na_col_to_ignore=[], 
                    dtype=None,
                    beta=None):
        """ 
        The function encodes the test file into SDV file. 
  
        Parameters
        ----------
        data_file : str, required
            The test file as CSV or NPY which needs to be converted.
        encoder_file : str, required
            The file that should be used for encoding this file.
        fix_na_values: boolean, optional
            Boolean variable which imputes the rows that have NA values in them (default is False.
        na_col_to_ignore: list, optional
            If the parameter "fix_na_values" is set to True, this list of columns will be ignored from imputation (default is empty list).
        dtype: dictionary, optional
            If you want to specify which column should be treated as continuous and which one as numeric, you can use Pandas' dtype dictioary for each column (default is None).
        beta: boolean, optional
            Use binary imputation rather than categorical for categorical columns (default is False).
          
        Outputs
        -------
        SDV file:
            The converted sdv file of the original test file provided with original name appended with "_sdv".
        """

        bar = IncrementalBar('Encoding', max=8)
        # Open and read the data file
        bar.next()
        df_raw = self.__read_data(data_file, dtype)
        bar.next()
        bar.next()
        bar.next()

        if fix_na_values:
            # Fix the NA values
            df_raw = self.__fix_na_values(df_raw, na_col_to_ignore)
            assert df_raw.isna().sum().sum() == 0

        enc_file = (
            encoder_file[:-4]
            if encoder_file.endswith(".csv")
            else encoder_file
        )
        bar.next()
        bar.next()
        bar.next()
        lims, mms, _, _ = self.__read_decoders(enc_file, "")
        df_converted, _, _ = self.__encode(df_raw, lims, mms, beta)
        bar.next()
        self.__save_files(df_converted, data_file[:-4])
        bar.finish()


class Decoder():
    """ 
    Decode files.

    The class provides functions to decode a data file and 
    produce the final synthetic data file.

    Available methods
    -----------------
        decode() : 
            The method generates a usable data file.
    """
    def __init__(self):
        pass

    def __read_decoders(self, prefix, npy_file):
        """
        Read the decoder files
        """
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
        """
        Read in the file
        """
        data = None
        if file_name.endswith(".csv") and dtype is not None:
            data = pd.read_csv(file_name, dtype = dtype)
        elif file_name.endswith(".csv"):
            data = pd.read_csv(file_name)
        elif file_name.endswith(".npy"):
            data = pd.DataFrame(np.load(file_name))

        # Check if file can be read
        if data is None:
            raise ValueError

        return data

    def __undo_categorical(self, col, lim):
        """
        Convert a categorical column to a continuous column
        """

        def cat_decode(x, limits):
            """
            Decoder for categorical data
            """
            for k, v in limits.items():
                if x <= float(k):
                    return v

        return col.apply(lambda x: cat_decode(x, lim))


    def __undo_numeric(self, col, min_col, max_col, discrete=None):
        """
        Normalize a numeric column
        """
        if discrete:
            return (((max_col - min_col) * col) + min_col).round().astype("int")
        return ((max_col - min_col) * col) + min_col

    def __decode(self, df_new, df_orig_cols, limits, min_max):
        """
        Decode the data from SDV format
        """
        df_new = pd.DataFrame(df_new, columns=df_orig_cols)
        for c in df_new.columns:
            if c in limits:
                df_new[c] = self.__undo_categorical(df_new[c], limits[c])
            else:
                df_new[c] = self.__undo_numeric(df_new[c], *min_max[c])

        return df_new

    def decode(self,
               data_file,
               npy_file):
        """ 
        The function decodes the file into a usable data file. 
  
        Parameters
        ----------
        data_file : str, required
            The file as CSV which needs needs to be decoded.
        npy_file : str, required
            The file that includes all description for decoding.
          
        Outputs
        -------
        Sythetic data file:
            The decoded synthetic file with the original file name appended with "_synthetic".
        """

        bar = IncrementalBar('Encoding', max=8)
        
        bar.next()
        lims, mm, cols, npy_new = self.__read_decoders(data_file[:-4], npy_file)
        bar.next()
        bar.next()
        bar.next()
        if not cols:
            # Open and read the data file
            df_raw = self.__read_data(data_file)
            cols = df_raw.columns

        bar.next()
        bar.next()
        bar.next()
        df_converted = self.__decode(np.clip(npy_new, 0, 1), cols, lims, mm)
        bar.next()
        df_converted.to_csv(data_file[:-4] + "_synthetic.csv", index=False)
        bar.finish()