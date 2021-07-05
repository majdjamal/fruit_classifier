
import numpy as np
import tensorflow.keras.applications.mobilenet as mob
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def MobileNet():

    dim = (224,224, 3)

    net = mob.MobileNet(
        input_shape=dim, alpha=1.0, depth_multiplier=1, dropout=0.001,
        include_top=True, weights='imagenet', pooling=None,
        classes=1000, classifier_activation='softmax')

    info = net.summary()

    return net

def ConvNet():

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='softmax'))

    info = model.summary()

    model.compile(optimizer='adam',
              loss= SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model
