
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Image Classification of fruits and vegetables')

parser.add_argument('--model', type = str, default='mobilenet',
	help='Model to train. Valid arguments are ["mobilenet", "efficientnetb0", "efficientnetb5", "inceptionv3"]. Default: "mobilenet"')

parser.add_argument('--generate_data', action = 'store_true', default=False,
	help='Generates training and testing data.')

parser.add_argument('--train',  action = 'store_true', default=False,
	help='Run the training script.')

parser.add_argument('--evaluate',  action = 'store_true', default=False,
	help='Run the evaluation script.')

parser.add_argument('--realtime',  action = 'store_true', default=False,
	help='Start Real-Time Image Classification.')

parser.add_argument('--predict',  action = 'store_true', default=False,
	help='Make a prediction of an image.')

parser.add_argument('--webcam_photos',  action = 'store_true', default=False,
	help='Evaluate network with webcam_photos.')


parser.add_argument('--max_brightness', type = float, default=0.95,
	help='Image Augumentation Argument. Maximum allowed brightness when generating the training set. Default: 0.95')

parser.add_argument('--min_brightness', type = float, default=0.6,
	help='Image Augumentation Argument. Minimum allowed brightness when generating the training set. Default: 0.6')

parser.add_argument('--max_noise', type = float, default=0.22,
	help='Image Augumentation Argument. Maximum allowed pixel distortion when generating the training set. Default: 0.22 ')
parser.add_argument('--min_noise', type = float, default=0.12,
	help='Image Augumentation Argument. Minimum allowed pixel distortion when generating the training set. Default: 0.12 ')


parser.add_argument('--transfer_learning', action ='store_true', default=False,
	help='Train with Transfer Learning')

parser.add_argument('--img_aug', type = bool, default=False,
	help='Bool, set "False" to construct networks with random initalization.')

parser.add_argument('--eta', type = float, default=1e-3,
	help='Learning Rate. Default: 0.001')

parser.add_argument('--roh', type = float, default=0.985,
	help='Momentum. Default: 0.985')

parser.add_argument('--wd', type = float, default=1e-5,
	help='Weight Decay. Default: 0.00001')

parser.add_argument('--NClasses', type = int, default=15,
	help='Number of classes in the dataset')

parser.add_argument('--dropout', type = float, default=2e-3,
	help='Dropout Rate. Default value is 0.002, used for MobileNet. Droput for EfficientNet should be 0.2.')

parser.add_argument('--epochs', type = int, default=700,
	help='Epochs. Default: 700')

parser.add_argument('--path', type = str, default='weights/mobilenet_transfer.ckpt',
	help='Path to pre trained model. Default: weights/mobilenet_transfer.ckpt')

parser.add_argument('--img_path', type = str, default='data/test_data/apple.png',
	help='Path to the image for prediction.')

args = parser.parse_args()
