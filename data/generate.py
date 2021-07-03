
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import time

start = time.time()

dataset = tfds.load(name="stanford_dogs")
sets = dataset.keys()
dataset = tfds.as_numpy(dataset)

i = 0

X_train = np.zeros((244, 244, 3, 10000))
y_train = np.zeros((1, 10000))

X_val = np.zeros((244, 244, 3, 4000))
y_val = np.zeros((1, 4000))

X_test = np.zeros((244,244, 3, 6580))
y_test = np.zeros((1,1, 6580))


dim = (244, 244)
ind = 0

for set in sets:

    for dog in dataset[set]:

        img = dog['image']
        lbl = dog['label']

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        if ind < 10000:
            curr_ind = ind

            X_train[:,:,:, curr_ind] = img
            y_train[:, curr_ind] = lbl

        elif ind < 14000:
            curr_ind = ind - 10000
            X_val[:,:,:, curr_ind] = img
            y_val[:, curr_ind] = lbl

        else:
            curr_ind = ind - 14000
            X_test[:,:,:, curr_ind] = img
            y_test[:, curr_ind] = lbl


        ind += 1

        print(ind)

np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)

np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

end = time.time()
print(end - start, 's')
