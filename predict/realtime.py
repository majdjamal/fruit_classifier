
import sys
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

ind_to_class = {
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


def RealTimeClassification(model, args):
	""" Real-Time Image Classification. Connects your webcam
	to a deep neural network and classifies which fruit it is seeing.
	:@param model: A trained network
	:@param args: Program arguments
	"""

	try:
		model.load_weights('weights/mobilenet_transfer.ckpt')
	except:
		raise ValueError(" \n Weights does not exist or does not match chosen network!")


	def process(img):
	    """ Pre-processes and classifies images. 
		:@param img: An image with shape (height, width, channels)
		:return preds: Top 4 prediction 
		:return labels: Labels for the top 4 predictions
	    """
	    dim = (224,224)

	    image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # CV_32F: 4-byte floating point (float)

	    pred = model.predict(image.reshape(1,224,224,3))

	    uncover_preds = np.argsort(-1*pred[0,:])[:4]

	    preds = []
	    labels = []

	    for i in uncover_preds:

	        preds.append(pred[0][i])
	        labels.append(ind_to_class[i])

	    return preds, labels	
	      

	def application():
	    """ Start a Matplotlib GUI. Captures photographs from users webcam, preprocess the image,
	    send it to the classifier, and update data in the GUI. 
	    """

	    print('=-=-=-=- Starting Application -=-=-=-= \n')	

	    camera = cv2.VideoCapture(0)

	    def closeApp(var):
	        print('\x1b[31m=-=-=-=- Program is stopped! -=-=-=-= \x1b[39m')
	        sys.exit(1)

	    ##
	    ##	Initializing GUI data
	    ##
	    y_pos = np.arange(4)
	    img = np.zeros((224,224, 3))
	    preds = [1,1,1,1]
	    labels = ['-','-','-','-']

	    ##
	   	## GUI Setup
	    ##
	    mpl.rcParams['toolbar'] = 'None'
	    fig, axs = plt.subplots(2, figsize=(10.5,6.5))
	    fig.patch.set_facecolor('lightsteelblue')
	    axs[0].set_title('Real-Time Image Classification  \n')

	    # Plot image on GUI
	    plotted_image = axs[0].imshow(img)
	    axs[0].axis('off')
	    
	    # Plot predictions in a horizontal bar chart
	    bars = axs[1].barh(y_pos, preds, align='center', color = '#4169E1')
	    lbls = axs[1].set_yticklabels(labels)
	    
	    ##
	    ##	Prediction bar color scheme
	    ##
	    bars[0].set_color('#193f6e')
	    bars[1].set_color('#3b6ba5')
	    bars[2].set_color('#72a5d3')
	    bars[3].set_color('#b1d3e3')
	    axs[1].set_facecolor('#e1ebec')
	    
	    ##
	   	## GUI Setup
	    ##
	    axs[1].invert_yaxis()
	    axs[1].set_yticks(y_pos)
	    axs[1].tick_params(axis='both', which='major', labelsize=14)
	    axnext = plt.axes([0.8, 0.9, 0.17, 0.075])
	    btnClose = Button(axnext, 'Close', color='#8b74bd')
	    btnClose.on_clicked(closeApp)
	    plt.ion()
	    plt.show()

	    # Real-time image classification in 3 steps
	    while True:

	        plt.pause(0.05)

	        return_value, image = camera.read()	# (1) captures a photo
	        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 	        
	        preds, labels = process(image) # (2) sends the image to the classifier
	        
	        ##
	        ## (3) Update data on the GUI
	        ##
	        plotted_image.set_data(image)

	        for bar, w in zip(bars, preds):
	            bar.set_width(w)

	        axs[1].set_yticklabels(labels)


	application()
