
import numpy as np
import tensorflow.keras.applications.mobilenet as mob
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def MobileNetModule():

    dim = (224,224, 3)

    net = mob.MobileNet(
        input_shape=dim, alpha=1.0, depth_multiplier=1, dropout=0.002,
        include_top=True, weights=None, pooling=None,
        classes=120, classifier_activation='softmax')

    info = net.summary()
    print(info)
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


def MobileNet():

    depth_multiplier = 1

    model = models.Sequential()
    model.add(layers.Input((224,224, 3)))

    model.add(layers.Conv2D(32, (3, 3), padding ='same', use_bias = False, strides = (2,2), name='conv1'))
    model.add(layers.BatchNormalization(axis = -1, name='conv1_bn'))
    model.add(layers.ReLU(6., name='conv1_relu'))

    # Depthwise Conv Block  1

    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (1,1), use_bias = False, name='conv_dw_1'))

    model.add(layers.BatchNormalization(axis = -1, name='conv_dw_1a_bn'))

    model.add(layers.ReLU(6., name='conv_dw_1a_relu'))
    model.add(layers.Conv2D(64, (1,1),
        padding='same', use_bias = False, strides = (1,1), name='conv_pw_1'))

    model.add(layers.BatchNormalization(axis = -1, name='conv_dw_1b_bn'))
    model.add(layers.ReLU(6., name='conv_dw_1b_relu'))


    # Depthwise Conv Block  2
    model.add(layers.ZeroPadding2D(((0, 1), (0, 1))))
    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (2,2), use_bias = False, name='conv_dw_2'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_2a_relu'))
    model.add(layers.Conv2D(128, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_2_relu'))


    # Depthwise Conv Block  3

    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (1,1), use_bias = False, name='conv_dw_3'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_3a_relu'))
    model.add(layers.Conv2D(128, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_3_relu'))

    # Depthwise Conv Block  4
    model.add(layers.ZeroPadding2D(((0, 1), (0, 1))))
    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (2,2), use_bias = False, name='conv_dw_4'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_4a_relu'))
    model.add(layers.Conv2D(256, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_4_relu'))


    # Depthwise Conv Block  5

    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (1,1), use_bias = False, name='conv_dw_5'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_5a_relu'))
    model.add(layers.Conv2D(256, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_5_relu'))


    # Depthwise Conv Block  6
    model.add(layers.ZeroPadding2D(((0, 1), (0, 1))))
    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (2,2), use_bias = False, name='conv_dw_6'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_6a_relu'))
    model.add(layers.Conv2D(512, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_6_relu'))


    # Depthwise Conv Block  7

    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (1,1), use_bias = False, name='conv_dw_7'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_7a_relu'))
    model.add(layers.Conv2D(512, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_7_relu'))


    # Depthwise Conv Block  8

    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (1,1), use_bias = False, name='conv_dw_8'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_8a_relu'))
    model.add(layers.Conv2D(512, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_8_relu'))


    # Depthwise Conv Block  9

    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (1,1), use_bias = False, name='conv_dw_9'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_9a_relu'))
    model.add(layers.Conv2D(512, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_9_relu'))


    # Depthwise Conv Block  10

    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (1,1), use_bias = False, name='conv_dw_10'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_10a_relu'))
    model.add(layers.Conv2D(512, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_10_relu'))


    # Depthwise Conv Block  11

    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (1,1), use_bias = False, name='conv_dw_11'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_11a_relu'))
    model.add(layers.Conv2D(512, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_11_relu'))


    # Depthwise Conv Block  12

    model.add(layers.ZeroPadding2D(((0, 1), (0, 1))))
    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'valid', depth_multiplier = depth_multiplier,
        strides = (2,2), use_bias = False, name='conv_dw_12'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_12a_relu'))
    model.add(layers.Conv2D(1024, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_12_relu'))



    # Depthwise Conv Block  13

    model.add(layers.DepthwiseConv2D((3,3),
        padding = 'same', depth_multiplier = depth_multiplier,
        strides = (1,1), use_bias = False, name='conv_dw_13'))

    model.add(layers.BatchNormalization(axis = -1))

    model.add(layers.ReLU(6., name='conv_dw_13a_relu'))
    model.add(layers.Conv2D(1024, (1,1),
        padding='same', use_bias = False, strides = (1,1)))

    model.add(layers.BatchNormalization(axis = -1))
    model.add(layers.ReLU(6., name='conv_dw_13_relu'))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Reshape((1,1,1024)))
    model.add(layers.Dropout(1e-3))
    model.add(layers.Conv2D(120, (1,1), padding = 'same', name='conv_preds'))
    model.add(layers.Activation(activation = 'softmax'))

    info = model.summary()

    print(info)
    return model
