# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
# with 6000 images per class. There are 50000 training images and 10000 test images.

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ds_train = tfds.load('cifar10', split= 'train', as_supervised=True)
ds_test = tfds.load('cifar10', split = 'test', as_supervised=True)
print (ds_train)


ds_train_1 = ds_train.take(1)
for image, label in tfds.as_numpy(ds_train_1):
    print(type(image), type(label), label)
    print(image)
image_tr, label_tr = tfds.as_numpy(tfds.load('cifar10', split = 'train', as_supervised=True, batch_size=-1))

print(type(image_tr), image_tr.shape)
