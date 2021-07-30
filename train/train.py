
__author__ = 'Majd Jamal'

import numpy as np
import random
import matplotlib.pyplot as plt
from data.loaddata import LoadData
from models.mobilenet import MobileNetModule
from models.efficientnet import EfficientNetModule
from data.loaddata import LoadTest
from skimage.util import random_noise
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train(model, args):
	
  X_train, X_val, y_train, y_val = LoadData()
  
  ###
  ### Image Augmentation
  ###

  def noise_generator(img):
      """ Adds noise to images.
      :@param img: image array with shape (Height, Width, Depth)
      :@param ind: current iteration state
      :return img: image with noise 7 times of 10, otherwise does nothing to the image.
      """
      state = random.randint(1,4)

      if state == 1:
        img = random_noise(img, mode='s&p',amount=args.max_noise)
      elif state == 2:
        img = random_noise(img, mode='s&p',amount=args.min_noise)
      elif state == 3:
        img = random_noise(img, mode='poisson')

      return img

  generator = ImageDataGenerator(
    rotation_range = 90,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    brightness_range = [args.min_brightness, args.max_brightness],
    zoom_range = [0.5,1.0],
    preprocessing_function = noise_generator
    )

  if args.transfer_learning:
    saving_path = "data/result/training_weights/"+args.model+"_transfer"
  else:
    saving_path = "data/result/training_weights/"+args.model

  cp_callback = ModelCheckpoint(filepath=saving_path+"/cp-{epoch:04d}.ckpt",
                                save_weights_only=True, monitor= 'val_accuracy', save_best_only=True,
                                verbose=1)

  generator.fit(X_train)
  training_history = model.fit(generator.flow(X_train, y_train), epochs=args.epochs, 
                  validation_data=(X_val, y_val),
                  shuffle = True, callbacks=[cp_callback], 
                  )
  
  np.save(saving_path + '/history.npy', training_history.history)  # history1=np.load('history1.npy',allow_pickle='TRUE').item()