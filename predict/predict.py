
__author__ = 'Majd Jamal'

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import one_hot, argmax
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import confusion_matrix, recall_score, precision_score


ind_to_class: dict = {
0 : 'Green Apple',
1:'Red Apple',
2:'Banana',
3:'Carrots',
4: "Chili",
5: "Corn",
6:'Kiwi',
7:'Lemon',
8:'Orange',
9:'Peach',
10:'Pear',
11:'Raspberry',
12:'Strawberry',
13:'Tomato',
14:'Watermelon'}

def predict(model, args) -> None:
	""" Predicts a label for one image.
	:@param model: Deep Network, type: tensorflow.keras.Sequential
	:@param args: System arguments, type: argparse.ArgumentParser
	"""

	try:
		model.load_weights(args.path)
	except:
		raise ValueError(" \n Weights does not exist or does not match chosen network!")

	dim = (224,224)

	try:
		image = plt.imread(args.img_path)
	except:
		raise ValueError(' Image does not exist in the given directory!')

	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # CV_32F: 4-byte floating point (float)
	image = image[:,:,:3]

	pred = model.predict(image.reshape(1,224,224,3))

	uncover_preds = np.argsort(-1*pred[0,:])[:4]

	refined_preds = []

	for i in uncover_preds:

		refined_preds.append(ind_to_class[i] + ' : ' + str(pred[0][i]*100)[:3] + '%')

	classification = argmax(pred, 1)
	classification = np.array(classification)

	print(args.img_path, ': ', refined_preds[0], ' | ', refined_preds[1], ' | ', refined_preds[2], ' | ', refined_preds[3])

def evaluate(model, args) -> None:
	""" Evaluates a trained model on Test Data.
	This function computes F1Score, Recall,
	Precision, and a Confusion Matrix.
	:@param model: Deep Network, type: tensorflow.keras.Sequential
	:@param args: System arguments, type: argparse.ArgumentParser
	"""

	from data.loaddata import LoadData, LoadTest, LoadPhotos

	try:
		model.load_weights(args.path)
	except:
		raise ValueError('\n File does not exist. Provide the right path with the command --path *PATH-TO-WEIGHT*.')

	if args.webcam_photos:
		X_test, y_test = LoadPhotos()
	else:
		X_test, y_test = LoadTest()

	y_test_hot = one_hot(y_test, args.NClasses)

	test_loss, test_acc = model.evaluate(X_test, y_test)
	y_pred = model.predict(X_test)

	###
	###	F1Score
	###
	f1 = F1Score(num_classes=args.NClasses)
	f1.update_state(y_test_hot[:,0,:], y_pred)
	f1 = f1.result()
	f1.numpy()

	y_pred = argmax(y_pred,1)

	###
	### Recall
	###
	rc = recall_score(y_test, y_pred, average=None)

	###
	### Precision
	###
	pr = precision_score(y_test, y_pred, average=None)

	###
	### Confusion
	###
	confusion = confusion_matrix(y_test, y_pred)


	plt.imshow(confusion, cmap ='Blues')
	plt.ylabel('True Label')
	plt.yticks(np.arange(0,args.NClasses))
	plt.xticks(np.arange(0,args.NClasses))
	plt.xlabel('Predicted Label')
	plt.savefig('data/result/confusion_matrix')
	plt.close()

	print('=-=-=- Test Data Evaluation -=-=-=')

	print('\x1b[33m Accuracy: ' + str(round(test_acc, 3)*100) + '% \x1b[39m')

	print('F1Score: ', np.array(f1))

	print('Recall: ', rc)

	print('Precision: ', pr)

	print('Confusion Matrix are found in data/result/ directory.')

	print('=-=-=-=-=-=-=-=-=-=-=-=-=-=--=')
