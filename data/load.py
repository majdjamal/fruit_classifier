
import numpy as np


def LoadData():

    #X_train = np.load('X_train.npy')    # 14.29 GB
    X_val = np.load('data/X_val.npy')        # 5.72 GB
    #X_test = np.load('X_test.npy')      #  9.4 GB

    #y_train = np.load('y_train.npy')
    y_val = np.load('data/y_val.npy')
    #y_test = np.load('y_test.npy')

    return X_val, y_val
