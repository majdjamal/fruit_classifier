
import numpy as np

class Data:

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

def LoadData():

    X_train = np.load('data/X_train.npy')
    X_val = np.load('data/X_val.npy')
    X_test = np.load('data/X_test.npy')

    y_train = np.load('data/y_train.npy')
    y_val = np.load('data/y_val.npy')
    y_test = np.load('data/y_test.npy')

    data = Data(
    X_train = X_train.transpose([3, 0,1,2]),
    X_val = X_val.transpose([3, 0,1,2]),
    X_test = X_test.transpose([3, 0,1,2]),
    y_train = y_train.T, y_val = y_val.T, y_test = y_test.T)

    return data
