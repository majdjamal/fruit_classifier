
import time

s = time.time()

import numpy as np
import tensorflow as tf
from models.mobilenet import ConvNet, MobileNet
from data.load import LoadData
import matplotlib.pyplot as plt

def train():
    #model = ConvNet()
    mobilenet = MobileNet()

    data = LoadData()

    mobilenet.compile(optimizer='adam',
                  loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = mobilenet.fit(data.X_train, data.y_train, epochs=1,
                    validation_data=(data.X_val, data.y_val))

    mobilenet.save_weights('./weights')

train()

e = time.time()

print(e - s, 's')
