
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
	print('=-=-=-=- Starting Data Generator -=-=-=-=')

	def noise_generator(img, ind):
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
	###	Initialize data structure
	###

	X_train = np.zeros((224, 224, 3, 9000))
	y_train = np.zeros((1, 9000))

	X_val = np.zeros((224, 224, 3, 2000))
	y_val = np.zeros((1, 2000))

	X_test = np.zeros((224,224, 3, 2000))
	y_test = np.zeros((1, 2000))

	dim = (224, 224)

	###
	###	Organize data
	###
	builder = tfds.ImageFolder('data/FRUITS/')
	ds = builder.as_dataset(split='train', shuffle_files=True)
	ds = tfds.as_numpy(ds)


	ind = 0

	#tqdm (range (100), desc="Loading..."):
	"""
	for fruit in ds:

	  img = fruit['image']
	  lbl = fruit['label']
	"""

	for fruit in tqdm (ds, total=13000, desc="Loading..."):

	  img = fruit['image']
	  lbl = fruit['label']

	  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	  img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	  if ind < 9000:

	    #img = noise_generator(img, ind)

	    X_train[:,:,:, ind] = img
	    y_train[:, ind] = lbl

	  elif ind < 11000:
	    curr_ind = ind - 9000
	    X_val[:,:,:, curr_ind] = img
	    y_val[:, curr_ind] = lbl

	  elif ind < 13000:
	    curr_ind = ind - 11000
	    X_test[:,:,:, curr_ind] = img
	    y_test[:, curr_ind] = lbl
	  
	  else:
	  	break
	  
	  ind += 1
	  
	print('=-=-=-=- Saving data -=-=-=-=')   
	
	np.save('data/training_data/X_train.npy', X_train)
	np.save('data/training_data/X_val.npy', X_val)
	np.save('data/test_data/X_test.npy', X_test)

	np.save('data/training_data/y_train.npy', y_train)
	np.save('data/training_data/y_val.npy', y_val)
	np.save('data/test_data/y_test.npy', y_test)

	print('=-=-=-=- Data generator complete! -=-=-=-=')
