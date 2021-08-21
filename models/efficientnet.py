
__author__ = 'Majd Jamal'

import ssl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB5
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow_addons.optimizers import SGDW

ssl._create_default_https_context = ssl._create_unverified_context

def EfficientNetModule(args) -> Sequential:
	""" Initializes and compiles EfficientNet.
	:@param args: Network arguments
	:return model: A compiled network
	"""

	dim = (224,224, 3)

	if args.transfer_learning:

		model = Sequential()

		if network == 'B0':

			effie = EfficientNetB0(
			    include_top=False,
			    weights="imagenet",
			    input_shape=dim,
			    classes=args.NClasses
			)

		elif network == 'B5':

			effie = EfficientNetB5(
			    include_top=False,
			    weights="imagenet",
			    input_shape=dim,
			    classes=args.NClasses
			)

		model.add(effie)
		model.add(GlobalAveragePooling2D())
		model.add(Dropout(args.dropout))
		model.add(Dense(args.NClasses, activation='softmax'))

	else:

		if network == 'B0':

			model = EfficientNetB0(
			    include_top=True,
			    weights=None,
			    input_shape=dim,
			    classes=args.NClasses,
			    classifier_activation="softmax",
				)

		elif network == 'B5':

			model = EfficientNetB5(
			    include_top=True,
			    weights=None,
			    input_shape=dim,
			    classes=args.NClasses,
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
