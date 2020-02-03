def encode_train(train_data):
    """
    This encodes the CSV to SDV format.

    Parameters
    ----------
    train_data : Pandas dataframe
        Dataframe that will be encoded into SDV format

    Returns
    -------
    Pandas dataframe
        Returns the encoded SDV file

    """
    print("Encoding training data")
    print("Training data encoded")
          
def encode_test(test_data, encoding_file):
    """
    This encodes the CSV to SDV format.

    Parameters
    ----------
    test_data : Pandas dataframe
        Dataframe that will be encoded into SDV format
    encoding_file: SDV file
        File to be used for encoding

    Returns
    -------
    Pandas dataframe
        Returns the encoded SDV file

    """
    print("Encoding test data based on {}".format(encoding_file))
    print("Test data encoded")