
__author__ = 'Majd Jamal'


import ssl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow_addons.optimizers import SGDW
import tensorflow.keras.layers as layers
from keras.engine import training
from keras.layers import VersionAwareLayers

ssl._create_default_https_context = ssl._create_unverified_context



def MobileNetFromScratch(NClasses = 15, dropout = 0.001):
	""" Re-implements the original MobileNet from scratch. 
	"""
	layers = VersionAwareLayers()

	def DepthwiseConv(inputs, NFilters, strides = (1,1)):

		if strides == (1,1):
			x = layers.ZeroPadding2D(((0, 1), (0, 1)))(inputs)
			padding = 'same'
		else:
			x = inputs 
			padding = 'valid'

		x = layers.DepthwiseConv2D((3,3), padding = padding, strides = strides, use_bias = False)(x)
		x = layers.BatchNormalization()(x)
		x = layers.ReLU(6.)(x)
		x = layers.Conv2D(NFilters, (1,1), padding = 'same', use_bias=False, strides = (1,1))(x)
		x = layers.BatchNormalization()(x)
		x = layers.ReLU(6.)(x)

		return x

	inp = layers.Input(shape = (224,224, 3))

	x = layers.Conv2D(32, (3,3), padding = 'same', use_bias = False, strides = (1,1))(inp)

	x = layers.BatchNormalization(axis = -1)(x)

	x = layers.ReLU(6.)(x)

	x = DepthwiseConv(x, 64)	# 1

	x = DepthwiseConv(x, 128, strides = (2,2)) # 2
	x = DepthwiseConv(x, 128) # 3


	x = DepthwiseConv(x, 256, strides = (2,2))	#  4
	x = DepthwiseConv(x, 256) # 5


	x = DepthwiseConv(x, 512, strides = (2,2)) #6
	
	x = DepthwiseConv(x, 512) # 7	
	x = DepthwiseConv(x, 512) # 8	
	x = DepthwiseConv(x, 512) # 9	
	x = DepthwiseConv(x, 512) # 10	
	x = DepthwiseConv(x, 512) # 11	



	x = DepthwiseConv(x, 1024, strides = (2,2)) # 12
	
	x = DepthwiseConv(x, 1024) # 13

	x = layers.GlobalAveragePooling2D()(x)

	x = layers.Reshape((1, 1, 1024))(x)
	x = layers.Dropout(dropout)(x)
	
	x = layers.Conv2D(NClasses, (1,1), padding = 'same')(x)
	x = layers.Reshape((NClasses, ))(x)
	x = layers.Activation(activation = 'softmax')(x)

	model = training.Model(inp, x)

	model.summary()

	return model


def MobileNetModule(args, transfer_learning = False, NClasses = 15):
	""" Initializes and compiles MobileNetV1.
	:@param transfer_learning: Type Bool. True to return network suited for transfer learning.
	:@param NClasses: Number of labels
	:@param args: Program arguments
	:return model: A compiled network
	"""
	dim = (224,224, 3)

	if args.transfer_learning:
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

		model.add(mobile)	#Add MobileNet
		model.add(GlobalAveragePooling2D()) #input_shape=effie.output_shape[1:]))
		model.add(Reshape((1, 1, 1024)))
		model.add(Dropout(args.dropout))
		model.add(Conv2D(NClasses, (1, 1)))
		model.add(Reshape((NClasses,)))
		model.add(Activation(activation = 'softmax'))

	else:

		model = MobileNetFromScratch()

	opt = SGDW(
		weight_decay = args.wd,
		learning_rate = args.eta,
		momentum = args.roh)

	model.compile(
		optimizer = opt,
		loss = SparseCategoricalCrossentropy(),
		metrics = ['accuracy'])

	return model
