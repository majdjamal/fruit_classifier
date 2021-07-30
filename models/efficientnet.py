
__author__ = 'Majd Jamal'

import ssl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB5
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow_addons.optimizers import SGDW

def EfficientNetModule(args, transfer_learning = False, network = 'B0', NClasses = 14):
	""" Initializes and compiles MobileNetV1. 
	:@param transfer_learning: Type Bool. True to return network suited for transfer learning.
	:@param NClasses: Number of labels
	:@param args: Program arguments
	:return model: A compiled network
	"""

	dim = (224,224, 3)

	if transfer_learning:

		model = Sequential()

		if network == 'B0':
			effie = EfficientNetB0(
			    include_top=False,
			    weights="imagenet",
			    input_shape=dim,
			    classes=NClasses
			)

		elif network == 'B5':

			effie = EfficientNetB5(
			    include_top=False,
			    weights="imagenet",
			    input_shape=dim,
			    classes=NClasses
			)

		model.add(effie)

		model.add(GlobalAveragePooling2D()) #input_shape=effie.output_shape[1:]))
		model.add(Dropout(args.dropout))
		model.add(Dense(NClasses, activation='softmax'))
		
	else:
		
		if network == 'B0':

			model = EfficientNetB0(
			    include_top=True,
			    weights=None,
			    input_shape=dim,
			    classes=NClasses,
			    classifier_activation="softmax",
				)

		elif network == 'B5':

			model = EfficientNetB5(
			    include_top=True,
			    weights=None,
			    input_shape=dim,
			    classes=NClasses,
			    classifier_activation="softmax",
				)


	model.summary()
	
	opt = SGDW(
		weight_decay = args.wd,
		learning_rate = args.eta,
		momentum = args.roh)

	model.compile(optimizer = opt,
	        loss = SparseCategoricalCrossentropy(),
	        metrics = ['accuracy'])

	return model