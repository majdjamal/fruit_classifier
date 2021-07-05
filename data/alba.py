
import cv2
import numpy as np
import matplotlib.pyplot as plt

def getImage():

    dim = (224, 224)
    img = plt.imread('data/alba.png') #, cv2.IMREAD_COLOR)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # CV_32F: 4-byte floating point (float)
    img = img[:,:,:3]

    return img
