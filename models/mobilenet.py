
__author__ = 'Majd Jamal'


import ssl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow_addons.optimizers import SGDW

ssl._create_default_https_context = ssl._create_unverified_context


def MobileNetModule(args, transfer_learning = False, NClasses = 14):
	""" Initializes and compiles MobileNetV1. 
	:@param transfer_learning: Type Bool. True to return network suited for transfer learning.
	:@param NClasses: Number of labels
	:@param args: Program arguments
	:return model: A compiled network
	"""
	dim = (224,224, 3)

	if transfer_learning:
		model = Sequential()

		mobile = MobileNet(
			input_shape=dim, 
			alpha=1.0, 
			depth_multiplier=1, 
			dropout=args.dropout,
			include_top=False,
			weights='imagenet', 
			pooling=None,
			classes=NClasses
		)

		model.add(mobile)
		model.add(GlobalAveragePooling2D()) #input_shape=effie.output_shape[1:]))
		model.add(Reshape((1, 1, 1024)))
		model.add(Dropout(args.dropout))
		model.add(Conv2D(NClasses, (1, 1)))
		model.add(Reshape((NClasses,)))
		model.add(Activation(activation = 'softmax'))

	else:

		model = MobileNet(	
			input_shape=dim, 
			alpha=1.0, 
			depth_multiplier=1, 
			dropout=args.dropout,
			include_top=True, 
			weights=None, 
			pooling=None,
			classes=14, 
			classifier_activation='softmax')

	model.summary()

	opt = SGDW(
		weight_decay = args.wd,
		learning_rate = args.eta,
		momentum = args.roh)

	model.compile(
		optimizer = opt,
		loss = SparseCategoricalCrossentropy(),
		metrics = ['accuracy'])

	return model
