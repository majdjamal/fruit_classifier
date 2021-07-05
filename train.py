
import numpy as np
import tensorflow as tf
from models.mobilenet import ConvNet
from data.load import LoadData


def train():
    model = ConvNet()

    X, y = LoadData()
    X = X.transpose([3, 0,1,2])
    y = y.T
    print(X.shape)
    print(y.shape)
    history = model.fit(X, y, epochs=10,
                    validation_data=(X, y))


train()
