
import time

s = time.time()

import numpy as np
import tensorflow as tf
from models.mobilenet import ConvNet, MobileNet, MobileNetModule
from data.load import LoadData
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

def train():
    #model = ConvNet()
    mobilenet = MobileNetModule()

    data = LoadData()

    opt = tfa.optimizers.AdamW(learning_rate=0.045, weight_decay=0.00004)

    mobilenet.compile(optimizer=opt,
                  loss= tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = mobilenet.fit_generator(data.X_train, data.y_train, epochs=300,
                    validation_data=(data.X_val, data.y_val))

    mobilenet.save_weights('./weights')

train()

e = time.time()

print(e - s, 's')
