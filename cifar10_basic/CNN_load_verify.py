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

model = tf.keras.models.load_model('cifar10_CNN.keras')

image_tr, label_tr = tfds.as_numpy(tfds.load('cifar10', split = 'train', as_supervised=True, batch_size=-1))
image_te, label_te = tfds.as_numpy(tfds.load('cifar10', split='test', as_supervised=True, batch_size=-1))

model.summary()


loss, acc = model.evaluate(image_te, label_te, verbose=2)
print('model_loss: {}, model_acc: {}'.format(loss, acc))

