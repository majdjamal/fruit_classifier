
__author__ = 'Majd Jamal'

import numpy as np
from data.loaddata import LoadData, LoadTest
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train(model, args):
  """ Train the network.
  :@param model: the network, in TensorFlow build format.
  :@param args: arguments from the parser
  """
  print('=-=-=-=-=- Network Arguments -=-=-=-=-=')
  print(args)
  X_train, X_val, y_train, y_val = LoadData()

  if args.fine_tune:
  
    try:
      model.load_weights(args.path)
    else:
      raise ValueError(" \n Weights does not exist or does not match chosen network!")

  # Image Augmentation
  generator = ImageDataGenerator(
    rotation_range = 90,
    vertical_flip = True,
    horizontal_flip = True,
    width_shift_range=0.1,
    dtype = float)

  def scheduler(epoch: int, lr: float) -> float:
    """ Learning rate decay
    """
    return lr * (1/(1 + 0.04 * epoch))

  sch_callback = LearningRateScheduler(scheduler)

  if args.transfer_learning:
    saving_path = "data/result/training_weights/"+args.model+"_transfer"
  else:
    saving_path = "data/result/training_weights/"+args.model


  cp_callback = ModelCheckpoint(filepath = saving_path+"/cp-{epoch:04d}.ckpt",
                                save_weights_only = True, 
                                monitor = 'val_accuracy', 
                                save_best_only = True,
                                verbose = 1
                                )

  training_history = model.fit(
    generator.flow(X_train, y_train[:,0], batch_size = 32), 
    epochs = args.epochs,
    validation_data = (X_val, y_val),
    callbacks = [cp_callback, sch_callback] if args.scheduler else [cp_callback]
    )

  np.save(saving_path + '/history.npy', training_history.history)  # history1=np.load('history1.npy',allow_pickle='TRUE').item()
