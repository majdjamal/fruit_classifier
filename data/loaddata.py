
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt


def LoadData():
    """ Loads data for training, i.e. training and validation set.
    """
    X_train = np.load('data/training_data/X_train.npy')
    X_val = np.load('data/training_data/X_val.npy')


    y_train = np.load('data/training_data/y_train.npy')
    y_val = np.load('data/training_data/y_val.npy')

    return X_train.transpose([3, 0,1,2]), X_val.transpose([3, 0,1,2]), y_train.T, y_val.T


def LoadTest():
    """ Loads data for evaluation, i.e. the test set.
    """
    X_test = np.load('data/test_data/X_test.npy')
    y_test = np.load('data/test_data/y_test.npy')

    return X_test.transpose([3, 0,1,2]), y_test.T
