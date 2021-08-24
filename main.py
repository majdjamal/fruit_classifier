
__author__ = 'Majd Jamal'

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.pars import args

# Removes Tensorflow warnings, used only when demonstrating the project results.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def setup_network():
	""" Setup and returns a deep network based on user input.
	"""
	if 'mobile' in args.model.lower():
		from models.mobilenet import MobileNetModule
		model = MobileNetModule(args = args)
		print('\n =-=-=-=- MobileNetV1 Loaded -=-=-=-= \n')

	elif 'b0' in args.model.lower():
		from models.efficientnet import EfficientNetModule
		model = EfficientNetModule(network_type = 'B0', args = args)
		print('\n =-=-=-=- EfficientNetB0 Loaded -=-=-=-= \n')

	elif 'b5' in args.model.lower():
		from models.efficientnet import EfficientNetModule
		model = EfficientNetModule(network_type = 'B5', args = args)
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
	:@param model: Deep Network, type: tensorflow.keras.Sequential
	:@param args: System arguments, type: argparse.ArgumentParser
	"""
	if args.generate_data:

		from data.gendata import GenerateData
		print('\n =-=-=-=- Loading data generator -=-=-=-= \n')
		GenerateData(args)

	elif args.train:

		from train.train import train
		print('\n =-=-=-=- Loading training script -=-=-=-= \n')
		train(model, args)

	elif args.evaluate:

		from predict.predict import evaluate
		print('\n =-=-=-=- Loading evaluation script -=-=-=-= \n')
		evaluate(model, args)

	elif args.realtime:
		import warnings
		warnings.filterwarnings("ignore")

		from predict.realtime import RealTimeClassification
		print('\n =-=-=-=- Loading Real-Time Image Classification script -=-=-=-= \n')
		RealTimeClassification(model, args)

	elif args.predict:
		from predict.predict import predict
		print('\n =-=-=-=- Loading prediction script -=-=-=-= \n')
		predict(model, args)

	else:
		raise ValueError('No operation were given! Please, specify what you want to do. For example, train a network with --train.')


def main() -> None:

	model = setup_network()
	setup_process(model, args)

if __name__ == "__main__":
	main()
