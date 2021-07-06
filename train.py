
import numpy as np
import tensorflow as tf
from models.mobilenet import ConvNet
from data.load import LoadData


def train():
    model = ConvNet()

    X, y = LoadData()
    X = X.transpose([3, 0,1,2])
    y = y.T

    history = model.fit(X[:10], y[:10], epochs=1,
                    validation_data=(X[:10], y[:10]))

    model.save_weights('./weights')

train()
