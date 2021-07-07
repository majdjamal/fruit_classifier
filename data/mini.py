
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

X_train = np.zeros((224, 224, 3, 5000))
y_train = np.zeros((1, 5000))

X_val = np.zeros((224, 224, 3, 250))
y_val = np.zeros((1, 250))

X_test = np.zeros((224,224, 3, 250))
y_test = np.zeros((1, 250))


dim = (224, 224)
ind = 0

for set in sets:

    for dog in dataset[set]:

        img = dog['image']
        lbl = dog['label']

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # CV_32F: 4-byte floating point (float)

<<<<<<< HEAD
        if ind < 5000:
=======
        if ind < 4000:
>>>>>>> aa27711c8567fc42027eaca60ad7f8644ce19add
            curr_ind = ind

            X_train[:,:,:, curr_ind] = img
            lmg = X_train[:,:,:, curr_ind]

            y_train[:, curr_ind] = lbl

<<<<<<< HEAD
        elif ind < 5250:
            curr_ind = ind - 5000
            X_val[:,:,:, curr_ind] = img
            y_val[:, curr_ind] = lbl

        elif ind < 5500:

            curr_ind = ind - 5250
=======
        elif ind < 4500:
            curr_ind = ind - 4000
            X_val[:,:,:, curr_ind] = img
            y_val[:, curr_ind] = lbl

        elif ind < 5000:

            curr_ind = ind - 4500
>>>>>>> aa27711c8567fc42027eaca60ad7f8644ce19add
            X_test[:,:,:, curr_ind] = img
            y_test[:, curr_ind] = lbl

        else:
            break

        ind += 1

        print(ind)

np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)

np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

end = time.time()

print((start - end)* (-1), 's')
