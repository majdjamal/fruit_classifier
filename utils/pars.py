
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Image Classification of fruits and vegetables')

parser.add_argument('--model', type = str, default='mobilenet',
	help='Deep Network to train. Valid arguments are ["mobilenet", "efficientnetb0", "efficientnetb5", "inceptionv3"]')

###
###	Main functions
###
parser.add_argument('--generate_data', type = bool, default=False,
	help='Bool, set "True" to generate_data.')

parser.add_argument('--train', type = bool, default=False,
	help='Bool, set "True" to train the network.')

parser.add_argument('--evaluate', type = bool, default=False,
	help='Bool, set "True" to generate_data.')

parser.add_argument('--realtime', type = bool, default=False,
	help='Bool, set "True" to run Real-Time Image Classification.')

parser.add_argument('--predict', type = bool, default=False,
	help='Bool, set "True" to generate_data.')

###
### Image Augumentation arguments
###
parser.add_argument('--max_brightness', type = float, default=0.95,
	help='Bool, set "True" to generate_data.')

parser.add_argument('--min_brightness', type = float, default=0.5,
	help='Bool, set "True" to generate_data.')

parser.add_argument('--max_noise', type = float, default=0.2,
	help='Bool, set "True" to generate_data.')
parser.add_argument('--min_noise', type = float, default=0.1,
	help='Bool, set "True" to generate_data.')

###
###	Training arguments
###
parser.add_argument('--transfer_learning', type = bool, default=True,
	help='Bool, set "False" to construct networks with random initalization.')

parser.add_argument('--eta', type = float, default=1e-3,
	help='Learning Rate')

parser.add_argument('--roh', type = float, default=0.99,
	help='Momentum')

parser.add_argument('--wd', type = float, default=1e-5,
	help='Weight Decay')

parser.add_argument('--NClasses', type = int, default=14,
	help='Weight Decay')

parser.add_argument('--dropout', type = float, default=2e-3,
	help='Dropout Rate. Default value is 0.002, used for MobileNet. Default for EfficientNet should be 0.2.')

parser.add_argument('--epochs', type = int, default=5e2,
	help='Training Cycles')


args = parser.parse_args()


