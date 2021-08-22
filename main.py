
__author__ = 'Majd Jamal'

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.pars import args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Removed excessive Tensorflow warnings

def setup_network():
	""" Setup and returns a deep network based on user input.
	"""
	if 'mobile' in args.model.lower():
		from models.mobilenet import MobileNetModule
		model = MobileNetModule(args = args)
		print('\n =-=-=-=- MobileNetV1 Loaded -=-=-=-= \n')

	elif 'b0' in args.model.lower():
		from models.efficientnet import EfficientNetModule
		model = EfficientNetModule(network = 'B0', args = args)
		print('\n =-=-=-=- EfficientNetB0 Loaded -=-=-=-= \n')

	elif 'b5' in args.model.lower():
		from models.efficientnet import EfficientNetModule
		model = EfficientNetModule(network = 'B5', args = args)
		print('\n =-=-=-=- EfficientNetB5 Loaded -=-=-=-= \n')

	elif 'resnet' in args.model.lower():
		from models.resnet import ResNet101Module
		model = ResNet101Module(args = args)
		print('\n =-=-=-=- ResNet101 Loaded -=-=-=-= \n')

	else:
		raise ValueError("Model does not exists! Valid models are [mobilenet, efficientnetb0, efficientnetb5]. Try again!")

	return model


def setup_process(model, args) -> None:
	""" Starts a process based on user input. E.g., train a network.
	"""

	if args.generate_data:

		from data.gendata import GenerateData
		GenerateData(args)

	elif args.train:

		from train.train import train
		train(model, args)
		print('\n =-=-=-=- Loading training script -=-=-=-= \n')

	elif args.evaluate:

		from predict.predict import evaluate
		evaluate(model, args)
		print('\n =-=-=-=- Loading evaluation script -=-=-=-= \n')

	elif args.realtime:
		import warnings
		warnings.filterwarnings("ignore")

		from predict.realtime import RealTimeClassification
		RealTimeClassification(model, args)
		print('\n =-=-=-=- Loading Real-Time Image Classification script -=-=-=-= \n')

	elif args.predict:
		from predict.predict import predict
		predict(model, args)
		print('\n =-=-=-=- Loading prediction script -=-=-=-= \n')
	else:
		raise ValueError('No operation were given! Please, specify what you want to do. For example, train a network with --train.')


def main() -> None:

	model = setup_network()
	setup_process(model, args)

if __name__ == "__main__":
	main()
