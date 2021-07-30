
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt

def plotter(preds, labels, img):
	""" Plotter for the Real-Time
	Image Classification.
	:@param preds: prediction stored in an array, shape (Npts,)
	:@param labels: label names ordered and stored in a list with shape (Npts,)
	:@param img: image taken by the webcam
	"""
	y_pos = np.arange(len(labels))

	fig, axs = plt.subplots(2)

	axs[0].imshow(img)
	axs[1].barh(y_pos, preds, align='center', color = '#4169E1')
	axs[1].set_yticklabels(labels)
	axs[1].invert_yaxis()
	axs[1].set_yticks(y_pos)
	axs[1].figure.set_size_inches(30, 14)
	axs[1].tick_params(axis='both', which='major', labelsize=20)
	plt.show(block=False)
	plt.pause(1)
	plt.close()