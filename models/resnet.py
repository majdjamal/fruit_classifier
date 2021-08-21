
__author__ = 'Majd Jamal'

import ssl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow_addons.optimizers import SGDW

ssl._create_default_https_context = ssl._create_unverified_context

def ResNet101Module(args) -> Sequential:
	""" Initializes and compiles InceptionV3.
	:@param transfer_learning: Type Bool. True to return network suited for transfer learning.
	:@param NClasses: Number of labels
	:@param args: Program arguments
	:return model: A compiled network
	"""
	dim = (224,224, 3)

	if args.transfer_learning:

		model = Sequential()

		res = ResNet101(
    	include_top=False,
    	weights='imagenet',
    	input_shape=dim,
    	classes=args.NClasses
		)

		model.add(res)
		model.add(GlobalAveragePooling2D())
		model.add(Dense(args.NClasses, activation = 'softmax'))

	else:

		model = ResNet101(
    	include_top=True,
    	weights=None,
    	input_shape=dim,
    	classes=args.NClasses
		)

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
