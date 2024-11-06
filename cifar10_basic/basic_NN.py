# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
# with 6000 images per class. There are 50000 training images and 10000 test images.

import tensorflow as tf
import tensorflow_datasets as tfds

import keras
from tensorflow.keras.models import Sequential                                    #type: ignore
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout        #type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D              #type: ignore
from tensorflow.keras.layers import BatchNormalization                            #type: ignore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading data using tensorflow_datasets into numpy arrays
image_tr, label_tr = tfds.as_numpy(tfds.load('cifar10', split = 'train', as_supervised=True, batch_size=-1))
image_te, label_te = tfds.as_numpy(tfds.load('cifar10', split='test', as_supervised=True, batch_size=-1))


#checking loaded data type and shape
print(type(image_tr), image_tr.shape)
print(type(image_te), image_te.shape)
print(type(label_tr), label_tr.shape)
print(label_tr[0])
# exit()

model = Sequential([
    keras.Input(shape = (3072,)),
    Dense(units = 200, activation = 'relu', name = 'layer1'),
    Dense(units = 100, activation = 'relu', name = 'layer2'),
    Dense(units = 50, activation = 'relu', name = 'layer3'),
    Dense(units = 25, activation ='relu', name='layer4'),
    Dense(units = 10, activation = 'linear', name = 'output_layer')
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics=['accuracy'])

x_train = np.reshape(image_tr, (np.shape(image_tr)[0],-1))
x_test = np.reshape(image_te, (np.shape(image_te)[0], -1))

print(x_train.shape)
print(x_train[0].shape)
print(x_test.shape)


x_train_nrm = np.array(x_train*1./255)
x_test_nrm = np.array(x_test*1./255)

print(x_train_nrm.shape)
print(x_test_nrm.shape)
print(x_test_nrm[0].shape)
# exit()

model.fit(x_train_nrm, label_tr, epochs = 20)



y_pred = model.predict(x_test_nrm[0].reshape(1, x_test_nrm[0].shape[0]))
print(y_pred)
y_predd = tf.nn.softmax(y_pred)
pred = np.argmax(y_predd)
print(pred)
print(label_tr[0])
