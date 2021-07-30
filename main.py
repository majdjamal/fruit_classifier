
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt
from utils.pars import args
from models.mobilenet import MobileNetModule
from models.efficientnet import EfficientNetModule

print(args.generate_data)

if args.model == 'mobilenet':
	args.path = 'data/weights/top_1/MobileNet/cp-0058.ckpt'
	model = MobileNetModule(transfer_learning = args.transfer_learning, args = args)

elif args.model == 'efficientnetb0':
	args.path = 'data/weights/top_1/EfficientNetB0/cp-0049.ckpt'
	model = EfficientNetModule(transfer_learning = args.transfer_learning, args = args)
	
elif args.model == 'efficientnetb5':
	args.path = 'data/weights/top_1/EfficientNetB5/cp-0030.ckpt'
	model = EfficientNetModule(transfer_learning = args.transfer_learning, 
  	network = 'B5', args = args)

else:
	raise ValueError("Model does not exists! Valid models are [mobilenet, efficientnetb0, efficientnetb5]. Try again!")


if args.generate_data:	#OK

	from data.gendata import GenerateData
	GenerateData(args)

elif args.train:

	from train.train import train
	train(model, args)

elif args.evaluate:	#OK

	from predict.predict import evaluate
	evaluate(model, args)

elif args.realtime:	#OK

	from predict.predict import RealTimeClassification
	RealTimeClassification(model, args)

elif args.predict:	#OK
	from predict.predict import predict
	predict(model, 'data/test_image/apple.png', args)
