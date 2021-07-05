
import time

start = time.time()

import numpy as np
import matplotlib.pyplot as plt
from data.load import LoadData
from models.mobilenet import MobileNet
from tensorflow.keras.applications import imagenet_utils
from data.alba import getImage

#img = getImage()
#img = img.reshape(1,224,224,3)
#print(img)

X, y = LoadData()

mobile = MobileNet()
#img = X[:,:,:,101].reshape(1,224,224,3)

#pred = mobile.predict(img)
#results = imagenet_utils.decode_predictions(pred)
#print(results)

#plt.imshow(X[:,:,:,101])
#plt.show()

end = time.time()
print(start - end, 's')
