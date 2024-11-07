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

#loading .keras model

model_keras = tf.keras.models.load_model('cifar10_CNN.keras')

image_tr, label_tr = tfds.as_numpy(tfds.load('cifar10', split = 'train', as_supervised=True, batch_size=-1))
image_te, label_te = tfds.as_numpy(tfds.load('cifar10', split='test', as_supervised=True, batch_size=-1))

model_keras.summary()

loss, acc = model_keras.evaluate(image_te, label_te, verbose=2)
print('model_loss: {}, model_acc: {}'.format(loss, acc))

sp = image_te[0].shape
validation_pos = []
validation_neg =[]
neg = 0

for i in range(len(image_te)):
    y_pred = model_keras.call(image_te[i].reshape(1,sp[0],sp[1],sp[2]))
    y_pred = tf.nn.softmax(y_pred)
    pred = np.argmax(y_pred)
    if pred == label_te[i]:
        validation_pos.append(pred)
    else :
        validation_neg.append([pred,label_te[i]])
        neg+=1
        print(label_te[i])
        print (pred)
        print('#',i)
print('number of test samples misclassified: ', neg)
model_h5 = tf.keras.models.load_model('cifar10_CNN_H5.h5')
model_h5.summary()
loss, acc = model_h5.evaluate(image_te, label_te, verbose=2)
print('model_loss_h5: {}, model_acc_h5: {}'.format(loss, acc))

model_by_wt = Sequential()

model_by_wt.add(Input(shape = image_tr[0].shape))

model_by_wt.add(Conv2D(32,(3,3), activation = 'relu', padding = 'same'))
model_by_wt.add(BatchNormalization())
model_by_wt.add(Conv2D(32,(3,3), activation = 'relu', padding = 'same'))
model_by_wt.add(BatchNormalization())
model_by_wt.add(MaxPooling2D((2,2)))

model_by_wt.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
model_by_wt.add(BatchNormalization())
model_by_wt.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
model_by_wt.add(BatchNormalization())
model_by_wt.add(MaxPooling2D((2,2)))

model_by_wt.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
model_by_wt.add(BatchNormalization())
model_by_wt.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
model_by_wt.add(BatchNormalization())
model_by_wt.add(MaxPooling2D((2,2)))

model_by_wt.add(Flatten())
model_by_wt.add(Dropout(0.2))
model_by_wt.add(Dense(1024, activation = 'relu'))
model_by_wt.add(Dropout(0.2))
model_by_wt.add(Dense(10, activation = 'linear'))

model_by_wt.summary()
model_by_wt.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

model_by_wt.load_weights('cifar10_basic/trained_weights/saved.weights.h5')


loss, acc = model_by_wt.evaluate(image_te, label_te, verbose=2)
print('model_loss_wt: {}, model_acc_wt: {}'.format(loss, acc))


