
import numpy as np
import tensorflow as tf
import tensorflow_datasets
import matplotlib.pyplot as plt

dataset, info = tensorflow_datasets.load(name="stanford_dogs", with_info=True)

for doggo in dataset['train'].take(5):

    dogs = doggo['image']
    print(type(dogs))
