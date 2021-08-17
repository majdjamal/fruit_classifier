
__author__ = 'Majd Jamal'

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib as mpl

def plotter(preds, labels, img):
	""" Plotter for the Real-Time
	Image Classification.
	:@param preds: prediction stored in an array, shape (Npts,)
	:@param labels: label names ordered and stored in a list with shape (Npts,)
	:@param img: image taken by the webcam
	"""
	def closeApp(var):
		print('\n =-=-=-=-=-=-=- \n Program is stopped!')
		sys.exit(1)


	y_pos = np.arange(len(labels))
	mpl.rcParams['toolbar'] = 'None'
	fig, axs = plt.subplots(2, figsize=(10.5,6.5))

	fig.patch.set_facecolor('lightsteelblue')
	
	axs[0].set_title('Real-Time Image Classification  \n')
	axs[0].imshow(img)
	axs[0].axis('off')
	bars = axs[1].barh(y_pos, preds, align='center', color = '#4169E1')
	
	bars[0].set_color('#193f6e')
	bars[1].set_color('#3b6ba5')
	bars[2].set_color('#72a5d3')
	bars[3].set_color('#b1d3e3')
	bars[4].set_color('#e1ebec')
	axs[1].set_facecolor('#f0e8dd')
	

	axs[1].set_yticklabels(labels)
	axs[1].invert_yaxis()
	axs[1].set_yticks(y_pos)
	axs[1].tick_params(axis='both', which='major', labelsize=14)
	axnext = plt.axes([0.8, 0.9, 0.17, 0.075])
	btnClose = Button(axnext, 'Close', color='#8b74bd')

	btnClose.on_clicked(closeApp)
	plt.ion()
	plt.show()

	return 
	#plt.show(block=False)
	#plt.pause(0.5)
	#plt.close()

#plotter([0.9,0.8,0.7,0.4, 0.3], ['Apple Green', 'B', 'C', 'D', 'E'], np.zeros((224,224,3)))

