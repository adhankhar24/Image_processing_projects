import tensorflow as tf
import tensorflow_datasets as tfds
import math
import keras
from tensorflow.keras.models import Sequential, Model                                    #type: ignore
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout               #type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D                     #type: ignore
from tensorflow.keras.layers import BatchNormalization                                   #type: ignore

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
print(label_tr)


# not flattening the input since we are utilizing convolution layers

model = Sequential()
model.add(Input(shape = image_tr[0].shape))

model.add(Conv2D(32,(3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'linear'))

model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

# creating callback for saving the model

checkpoint_path = 'cifar10_basic/training_cifar_cp/cp-{epoch:04d}.weights.h5'

batch_size = 32
n_batch = math.ceil(len(image_tr)/32)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path, 
    verbose = 1,
    save_weights_only=True,
    save_freq = 5*n_batch
)

model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(
    image_tr, label_tr, 
    epochs = 10, 
    batch_size = batch_size,
    callbacks = [cp_callback],
    verbose=0)

validation_pos = []
validation_neg = []
neg = 0
sp = image_te[0].shape

for i in range(len(image_te)):
    y_pred = model.call(image_te[0].reshape(1,sp[0],sp[1],sp[2]))
    y_pred = tf.nn.softmax(y_pred)
    pred = np.argmax(y_pred)
    if pred == label_te[i]:
        validation_pos.append(pred)
    else :
        validation_neg.append([pred,label_te[i]])
        neg+=1
print(neg)


# saving model

# saving in .keras format
model.save('cifar10_CNN.keras')

# saving in h5 format
model.save('cifar10_CNN_H5.h5')

#  manually saving weights
weight_path = 'cifar10_basic/trained_weights/saved.weights.h5'
model.save_weights(weight_path)