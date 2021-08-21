
import sys
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class Application:

	def __init__(self, model, args):

		self.model = model
		self.args = args

		self.axs = None	# prediction bars GUI, used to adjust labels
		self.bars = None # prediction bars
		self.plotted_image = None # Image on the GUI
		self.btnClose = None # Close button

		self.ind_to_class: dict = {
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


		self.dim: tuple = (224,224)


	def image_processing(self, img: np.ndarray) -> list:
		""" Pre-processes and classifies images. 
		:@param img: An image with shape (height, width, channels)
		:return preds: Top 4 prediction stored in a list
		:return labels: Labels for the top 4 predictions stored in a list
		"""

		image = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)

		image = cv2.normalize(image, None, 
			alpha=0, 
			beta=1, 
			norm_type=cv2.NORM_MINMAX, 
			dtype=cv2.CV_32F)

		pred = self.model.predict(image.reshape(1,224,224,3))

		uncover_preds = np.argsort(-1*pred[0,:])[:3]

		preds = []
		labels = []

		for i in uncover_preds:

			preds.append(pred[0][i])
			labels.append(self.ind_to_class[i])

		return preds, labels	
	
	def GUI(self):
		""" Setup and start the graphical user interface with Matplotlib.
		"""

		print('=-=-=-=- Starting Application -=-=-=-= \n')

		def closeApp(var):
			print('Been Here!')
			print('\x1b[31m=-=-=-=- Program is stopped! -=-=-=-= \x1b[39m')
			sys.exit(1)

		##
		##	Initializing GUI data
		##
		y_pos = np.arange(3)
		img = np.zeros((224,224, 3))
		preds = [1,1,1]
		labels = ['-','-','-']

		##
	   	## GUI Setup
		##
		mpl.rcParams['toolbar'] = 'None'
		fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6.5,6.5))


		fig.patch.set_facecolor('lightsteelblue')
		axs[0].set_title('Real-Time Image Classification  \n', fontweight="bold")

		# Plot image on GUI
		self.plotted_image = axs[0].imshow(img)
		axs[0].axis('off')
		

		# Plot predictions in a horizontal bar chart
		bars = axs[1].barh(y_pos, preds, align='center', color = '#4169E1')
		lbls = axs[1].set_yticklabels(labels)
		
		axs[1].spines['bottom'].set_color('lightsteelblue')
		axs[1].spines['top'].set_color('lightsteelblue') 
		axs[1].spines['right'].set_color('lightsteelblue')
		axs[1].spines['left'].set_color('lightsteelblue')
		axs[1].margins(0.01, 0.05)
		axs[1].set_facecolor('#e1ebec')

		##
		##	Prediction bar color scheme
		##
		bars[0].set_color('#193f6e')
		bars[1].set_color('#3b6ba5')
		bars[2].set_color('#72a5d3')
		
		self.bars = bars
		
		##
	   	## GUI Setup
		##
		axs[1].invert_yaxis()
		axs[1].set_yticks(y_pos)
		axs[1].set_xticks([0,0.5, 1])
		axs[1].set_xticklabels(['0%','50%','100%'])
		axs[1].tick_params(axis='both', which='major', labelsize=12)
		fig.subplots_adjust(hspace=0.05, right = 0.8, left = 0.2, bottom=0.15) # **

		axnext = plt.axes([0.45, 0.03, 0.1, 0.05])
		self.btnClose = Button(axnext, 'Close', color='#8b74bd')

		self.btnClose.on_clicked(lambda var: sys.exit(1))

		self.axs = axs

		plt.ion()
		plt.show()


	def start(self):

		self.GUI()

		camera = cv2.VideoCapture(0)

		# Real-time image classification in 3 steps
		while True:

			plt.pause(0.01)

			return_value, image = camera.read()	# (1) captures a photo
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 			
			preds, labels = self.image_processing(image) # (2) sends the image to the classifier
			
			##
			## (3) Update data on the GUI
			##
			self.plotted_image.set_data(image)

			for bar, w in zip(self.bars, preds):
			    bar.set_width(w)

			self.axs[1].set_yticklabels(labels)


def RealTimeClassification(model, args):
	""" Real-Time Image Classification. Connects your webcam
	to a deep neural network and classifies which fruit it is seeing.
	:@param model: A trained network
	:@param args: Program arguments
	"""

	try:
		model.load_weights(args.path)
	except:
		raise ValueError(" \n Weights does not exist or does not match chosen network!")

	app = Application(model, args)
	app.start()
