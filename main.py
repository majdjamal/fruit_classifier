
__author__ = 'Majd Jamal'

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.pars import args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Removed excessive Tensorflow warnings

if 'mobile' in args.model:
	from models.mobilenet import MobileNetModule
	model = MobileNetModule(transfer_learning = args.transfer_learning, args = args)

elif args.model == 'efficientnetb0':
	from models.efficientnet import EfficientNetModule
	model = EfficientNetModule(transfer_learning = args.transfer_learning, args = args)

elif args.model == 'efficientnetb5':
	from models.efficientnet import EfficientNetModule
	model = EfficientNetModule(transfer_learning = args.transfer_learning,
  	network = 'B5', args = args)

elif 'resnet' in args.model:
	from models.resnet import ResNet101Module
	model = ResNet101Module(transfer_learning = args.transfer_learning, args = args)

else:
	raise ValueError("Model does not exists! Valid models are [mobilenet, efficientnetb0, efficientnetb5]. Try again!")


if args.generate_data:

	from data.gendata import GenerateData
	GenerateData(args)

elif args.train:

	from train.train import train
	train(model, args)

elif args.evaluate:

	from predict.predict import evaluate
	evaluate(model, args)

elif args.realtime:
	import warnings
	warnings.filterwarnings("ignore")

	from predict.realtime import RealTimeClassification
	RealTimeClassification(model, args)

elif args.predict:
	from predict.predict import predict
	predict(model, 'data/test_data/apple.png', args)
