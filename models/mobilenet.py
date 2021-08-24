
__author__ = 'Majd Jamal'


import ssl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation,DepthwiseConv2D, BatchNormalization, ReLU
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow_addons.optimizers import SGDW
import tensorflow.keras.layers as layers
from keras.utils import data_utils

ssl._create_default_https_context = ssl._create_unverified_context

def MobileNetFromScratch(transfer_learning: bool = True, NClasses: int = 15, dropout: float = 0.001, alpha = 1., depth_multiplier = 1, config: bool = False):
	""" Re-implements the original MobileNet from scratch. This module took inspiration from:
	https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py#L80-L313
	:@param transfer_learning: Bool, true to upload imagenet weights
	:@param NClasses: Number of categories in the dataset
	:@param droput: Dropout regularization
	:retunr model: MobileNetV1
	"""

	model = Sequential()

	def ConvBlock(model: Sequential, NFilters: int = 32, _id: str = '1', strides: tuple = (1,1)) -> Sequential:
		""" Convolutional block. From Howards et al, (2017)
		Figure 3. Left.
		:@param model: Current architecture
		:@param NFilters: Number of filters in the point-wise convolutions
		:@param strides: Strides
		:return model: Model with added Conv block
		"""

		NFilters *= alpha
		NFilters = int(NFilters)

		model.add(layers.Conv2D(
			NFilters,
			(3,3),
			padding = 'same',
			use_bias = False,
			strides = strides,
			name = 'Conv_{}'.format(str(_id)))
		)

		model.add(layers.BatchNormalization(
			axis = -1,
			name = 'Conv_{}_BN'.format(str(_id)))
		)

		model.add(layers.ReLU(
			6.,
			name = 'Conv_{}_ReLU'.format(str(_id)))
		)

		return model

	def DWSBlock(model: Sequential, NFilters: int, _id: str, strides: tuple = (1,1)) -> Sequential:
		""" Depth-Wise Seperable Convolution Block.
		From Howards et al, (2017) Figure 3. Right.
		:@param model: Current architecture, type: tensorflow.keras.Sequential
		:@param NFilters: Number of filters in the point-wise convolutions
		:@param strides: Strides
		:return model: Model with added DWS block
		"""

		NFilters *= alpha
		NFilters = int(NFilters)

		if strides == (1,1):
			padding = 'same'

		else:
			model.add(layers.ZeroPadding2D(
				((0, 1), (0, 1)))
			)
			padding = 'valid'

		##
		##	Depth-Wise Convolutions
		##
		model.add(DepthwiseConv2D(
			(3,3),
			padding = padding,
			strides = strides,
			use_bias = False,
			depth_multiplier = depth_multiplier,
			name = 'DWS_{}'.format(str(_id)))
		)

		model.add(BatchNormalization(
			axis = -1,
			name = 'DWS_{}_BN'.format(str(_id)))
		)

		model.add(ReLU(
			6.,
			name = 'DWS_{}_ReLU'.format(str(_id)))
		)

		##
		##	Point-Wise Convolutions
		##
		model.add(Conv2D(
			NFilters,
			(1,1),
			padding = 'same',
			use_bias=False,
			strides = (1,1),
			name = 'POINT-W_{}'.format(str(_id)))
		)

		model.add(BatchNormalization(
			axis = -1,
			name = 'POINT-W_{}_BN'.format(str(_id)))
		)

		model.add(ReLU(
			6.,
			name = 'POINT-W_{}_ReLU'.format(str(_id)))
		)

		return model


	model.add(layers.Input(shape = (224,224, 3)))
	model = ConvBlock(model, strides = (2,2))
	model = DWSBlock(model, 64, _id = '1')
	model = DWSBlock(model, 128, strides = (2,2), _id = '2')
	model = DWSBlock(model, 128, _id = '3')
	model = DWSBlock(model, 256, strides = (2,2), _id = '4')
	model = DWSBlock(model, 256, _id = '5')
	model = DWSBlock(model, 512, strides = (2,2), _id = '6')
	model = DWSBlock(model, 512, _id = '7')

	if not config:

		model = DWSBlock(model, 512, _id = '8')
		model = DWSBlock(model, 512, _id = '9')
		model = DWSBlock(model, 512, _id = '10')
		model = DWSBlock(model, 512, _id = '11')
		model = DWSBlock(model, 1024, strides = (2,2), _id = '12')
		model = DWSBlock(model, 1024, _id = '13')

	if transfer_learning:

		model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)
		basePath = ('https://storage.googleapis.com/tensorflow/'
                    'keras-applications/mobilenet/')
		weights_path = basePath + model_name

		weights = data_utils.get_file(
          model_name, weights_path, cache_subdir='models')

		model.load_weights(weights)

	##
	##	Bottom layers
	##
	model.add(layers.GlobalAveragePooling2D())

	if not config:
		model.add(layers.Reshape((1, 1, int(1024*alpha))))
	else:
		model.add(layers.Reshape((1, 1, int(512*alpha))))

	model.add(layers.Dropout(dropout))
	model.add(layers.Conv2D(NClasses, (1,1), padding = 'same'))
	model.add(layers.Reshape((NClasses, )))
	model.add(layers.Activation(activation = 'softmax'))


	return model


def MobileNetModule(args) -> Sequential:
	""" Initializes and compiles MobileNetV1.
	:@param args: System arguments, type: argparse.ArgumentParser

	Experiment results are documented in table 1.

	-----------------------------------------------------------------
	|	      Model        |			  Accuracy					|
	|		               |	Random Init.  |   Transfer Learning |
	----------------------------------------------------------------|
	|	MobileNetV1		   | 		92.7%	  |        96.8%		|
	|	MobileNet Config   |        93.9%     |          -          |
	-----------------------------------------------------------------
	(Table 1. Performances of MobileNet on the fruit and vegetable dataset.
	Note. Transfer Learning for MobileNet Config is missing because there was
	not an available pre_trained model for the configured architecture.)
	"""

	model = MobileNetFromScratch(
	transfer_learning = args.transfer_learning,
	NClasses = args.NClasses,
	dropout = args.dropout,
	alpha = args.alpha,
	depth_multiplier = args.depth_multiplier,
  config = args.config)

	model.summary()

	opt = SGDW(
		momentum = args.roh,
		learning_rate = args.eta,
		weight_decay = args.wd)

	model.compile(
		optimizer = opt,
		loss = SparseCategoricalCrossentropy(),
		metrics = ['accuracy'])

	return model
