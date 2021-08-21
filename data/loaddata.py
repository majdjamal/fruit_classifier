
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt

def LoadData() -> np.ndarray:
    """ Loads data for training, i.e. training and validation set.
    :return X_train: Traning patterns, shape = (Npts, height, width depth)
    :return y_train: Traning targets, shape = (Npts, 1)
    :return X_val: Validation patterns, shape = (Npts, height, width depth)
    :return y_val: Validation targets, shape = (Npts, 1)
    """
    X_train = np.load('data/training_data/X_train.npy')
    X_val = np.load('data/training_data/X_val.npy')


    y_train = np.load('data/training_data/y_train.npy')
    y_val = np.load('data/training_data/y_val.npy')

    return X_train.transpose([3, 0,1,2]), X_val.transpose([3, 0,1,2]), y_train.T, y_val.T


def LoadTest() -> np.ndarray:
    """ Loads data for evaluation, i.e. the test set.
    :return X_test: Test patterns, shape = (Npts, height, width depth)
    :return y_test: Test targets, shape = (Npts, 1)
    """
    X_test = np.load('data/test_data/X_test.npy')
    y_test = np.load('data/test_data/y_test.npy')

    return X_test.transpose([3, 0,1,2]), y_test.T


def LoadPhotos() -> np.ndarray:
    """ Generates and loads webcam images, used to evaluate the real-time image classification app.
    :return X_webcam: Traning patterns, shape = (Npts, height, width depth)
    :return y_webcam: Traning targets, shape = (Npts, 1)
    """

    import tensorflow_datasets as tfds
    import cv2
    import tqdm

    dim = (224, 224)

    ###
    ### Open and read data folder
    ###
    builder = tfds.ImageFolder('data/webcam_photos/')
    ds_webcam = builder.as_dataset(split='train', shuffle_files=True)
    ds_webcam = tfds.as_numpy(ds_webcam)

    Npts = len(ds_webcam)

    ##
    ## Initialize data matricies
    ##
    X_webcam = np.zeros((224, 224, 3, Npts))
    y_webcam = np.zeros((1, Npts))

    ind = 0

    for fruit in ds_webcam:

      img = fruit['image']
      lbl = fruit['label']

      img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
      img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

      ## Store data in training set
      if ind < Npts:

        X_webcam[:,:,:, ind] = img
        y_webcam[:, ind] = lbl

      else:

        break

      ind += 1

    return X_webcam.transpose([3, 0,1,2]), y_webcam.T
