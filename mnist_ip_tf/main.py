import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_loader as mod1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


train_images = 'mnist_ip_tf/train-images.idx3-ubyte'
train_labels = 'mnist_ip_tf/train-labels.idx1-ubyte'
test_images = 'mnist_ip_tf/t10k-images.idx3-ubyte'
test_labels = 'mnist_ip_tf/t10k-labels.idx1-ubyte'

dataset = mod1.MnistDataloader(train_images, train_labels, test_images, test_labels)
(x_train, y_train), (x_test, y_test) = dataset.load_data()


print('x train shape = {}'.format(np.shape(x_train)))
print('x test shape = {}'.format(np.shape(x_test)))
print('y train shape = {}'.format(np.shape(y_train)))
print('y test shape = {}'.format(np.shape(y_test)))

# flattening the inputs

x_train_flat = np.reshape(x_train, (np.shape(x_train)[0], -1))
x_test_flat = np.reshape(x_test, (np.shape(x_test)[0],-1))

# Note that the -1 there is not anything magic: it is just a shortcut that means 
# “use all the leftover values of the remaining dimensions here”. 
# You could get the same result with this statement:
# x_train_flat = np.reshape(x_train, (np.shape(x_train)[0], np.shape(x_train)[1]*np.shape(x_train)[2]))

# (784, 60000)
print(np.shape(x_train_flat)) 
# (60000,)
print(np.shape(y_train))

print(np.shape(x_test_flat[0])) 
# exit()

# normalizing inputs
# x_train_flat = x_train_flat.astype(float)
# x_train_flat = 1./255

# x_test_flat = x_test_flat.astype(float)
# x_test_flat = 1./255

# print(dtype(x_train_flat))

model = Sequential([
    
        tf.keras.Input(shape = (784,)),
    Dense(units = 40, activation = 'relu', name = 'Layer1'),
    Dense(units = 20, activation = 'relu', name = 'Layer2'),
    Dense(units = 10, activation = 'linear', name = 'Layer3')
    
])

model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
x_train_flat = np.array(x_train_flat*1./255)
y_train = np.array(y_train)
x_test_flat = np.array(x_test_flat*1./255)
history = model.fit(x_train_flat, y_train, epochs=35)



# model accuracy on test set with used parameters 
# Model archteicture -> 784,40 -> 40,20 -> 20,10 (0-9: 10 classes)
# epochs = 40

l=0
for i in range(len(x_test)):
   y_pred = model.call(np.reshape(x_test_flat[i],(1,784)))
   y_pred = tf.nn.softmax(y_pred)
   y_predd = np.argmax(y_pred)
   if y_predd!= y_test[i]:
      l+=1
print (f"the model's accuracy on test data = {100. - (100.*l/10000)}%")

