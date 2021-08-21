
__author__ = 'Majd Jamal'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from skimage.util import random_noise
from skimage import exposure
from tqdm import tqdm

def GenerateData(args):
	""" Generates data and saves it in .npy files.
	:@param args: program arguments
	"""

	dim = (224, 224)

	print('=-=-=-=- Starting Data Generator -=-=-=-=')

	def noise_generator(img: np.ndarray, ind: int) -> np.ndarray:
	  """ Adds noise to images.
	  :@param img: image array with shape (Height, Width, Depth)
	  :@param ind: current iteration state
	  :return img: image with noise if requirements are meet.
	  """

	  if ind % 5 == 0:
	    img = random_noise(img, mode='s&p',amount=args.max_noise)
	  elif ind % 2 == 0:
	    img = random_noise(img, mode='s&p',amount=args.min_noise) * args.min_brightness
	  elif ind % 3 == 0:
	    img = random_noise(img, mode='poisson') * args.max_brightness

	  return img

	###
	###	Open and read data folder
	###
	builder = tfds.ImageFolder('data/FRUITS/')
	ds_train = builder.as_dataset(split='train', shuffle_files=True)
	ds_train = tfds.as_numpy(ds_train)

	ds_test = builder.as_dataset(split='test', shuffle_files=True)
	ds_test = tfds.as_numpy(ds_test)

	validation_split = 0.2

	Npts_train = 11343
	Npts_val = int(np.floor(Npts_train * validation_split))

	Npts_test = 3000

	##
	## Initialize data matricies
	##
	X_train: np.ndarray = np.zeros((224, 224, 3, Npts_train - Npts_val))
	y_train: np.ndarray = np.zeros((1, Npts_train - Npts_val))

	X_val: np.ndarray = np.zeros((224, 224, 3, Npts_val))
	y_val: np.ndarray = np.zeros((1, Npts_val))

	X_test: np.ndarray = np.zeros((224,224, 3, Npts_test))
	y_test: np.ndarray = np.zeros((1, Npts_test))


	def generate(ds, training_data: bool = False, Npts: int = None):
		""" Generates data. Reading images files and converts them to
		NumPy arrays.
		:@param ds: opened dataset in ImageFolder format.
		:@param training_data: bool, True if dataset is for training.
		:@param Npts: Number of points in the dataset.
		"""
		ind = 0

		for fruit in tqdm (ds, total=Npts, desc="Loading..."):

		  img = fruit['image']
		  lbl = fruit['label']

		  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		  img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		  if training_data:

		  	  ## Store data in training set
			  if ind < Npts_train - Npts_val:

			    img = noise_generator(img, ind)

			    X_train[:,:,:, ind] = img
			    y_train[:, ind] = lbl

			  ## Store data in validation set
			  elif ind < Npts_train:

			    curr_ind = ind - Npts_train

			    X_val[:,:,:, curr_ind] = img
			    y_val[:, curr_ind] = lbl


		  else:

		  	 ## Store data in test set
			  if ind < Npts_test:

			   X_test[:,:,:, ind] = img
			   y_test[:, ind] = lbl

		  ind += 1

	print('=-=-=-=- Generating training data -=-=-=-=')
	generate(ds_train, training_data = True, Npts = Npts_train)
	print('=-=-=-=- Generating testing data -=-=-=-=')
	generate(ds_test, Npts = Npts_test)


	print('=-=-=-=- Saving data -=-=-=-=')

	np.save('data/training_data/X_train.npy', X_train)
	print('=-=- Training Pattern Saved -=-=')
	np.save('data/training_data/X_val.npy', X_val)
	print('=-=- Validation Pattern Saved -=-=')
	np.save('data/test_data/X_test.npy', X_test)
	print('=-=- Test Pattern Saved -=-=')

	np.save('data/training_data/y_train.npy', y_train)
	print('=-=- Training Targets Saved -=-=')
	np.save('data/training_data/y_val.npy', y_val)
	print('=-=- Validation Targets Saved -=-=')
	np.save('data/test_data/y_test.npy', y_test)
	print('=-=- Testing Targets Saved -=-= \n')

	print('=-=-=-=- Data generator complete! -=-=-=-=')
