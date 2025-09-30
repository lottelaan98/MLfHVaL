# The CIFAR-10 dataset contains colour images of objects, each 32x32x3 pixels (for 
# the three colour channels). These have 10 categories (or classes) of object (airplane, 
# automobile, bird, cat, deer, dog, frog, horse, ship & truck) with 5,000 images in each, 
# making a total of 50,000 images in the training set (x_train, y_train), randomly 
# ordered with numerical labels for each (1=airplane, 2=automobile etc.). The test set 
# (x_test, y_test) contains 10,000 images ordered by their label. 

import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10, mnist
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)

(x_trainc, y_trainc), (x_testc, y_testc) = keras.datasets.cifar10.load_data() 
(x_trainm, y_trainm), (x_testm, y_testm) = keras.datasets.mnist.load_data()

# Prepare the test and training images by dividing their values by 255, storing the 
# result in variables x_train and x_test. The shapes of the image matrices are already 
# correct for input into Keras. Convert the training and test labels to categorical 
# variables, as for the handwritten digits in Learning-based Exercise One, storing the 
# result in variables y_train and y_test.

x_trainc = x_trainc.astype('float32') / 255
x_testc = x_testc.astype('float32') / 255
y_trainc = keras.utils.to_categorical(y_trainc, 10)
y_testc = keras.utils.to_categorical(y_testc, 10)

x_trainm = x_trainm.astype('float32') / 255
x_testm = x_testm.astype('float32') / 255
x_trainm = np.expand_dims(x_trainm, -1)  # Add channel dimension
x_testm = np.expand_dims(x_testm, -1)    # Add channel dimension
y_trainm = keras.utils.to_categorical(y_trainm, 10)
y_testm = keras.utils.to_categorical(y_testm, 10)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# output = (50000, 32, 32, 3) (50000, 10) (10000, 32, 32, 3) (10000, 10)

# Define the model using the convolutional network with dropout (from Learning-based 
# Exercise One’s ‘Deep convolutional networks’ topic (Questions 6-7)) as a template.  -For the first convolutional layer, add the arguments:  
# input_shape = (32, 32, 3), padding = “same” 
# to the layer_conv_2d call. - In the second convolutional layer, use 32 filters instead of 64 to reduce 
# computational load. -After max pooling and dropout layers, repeat these layers again (add conv, conv, 
# pool, dropout, after the existing conv, conv, pool, dropout). There is no need to 
# define input_shape here. -Flatten the result and link it to a larger fully-connected layer than before, using 512 
# units instead of 128, with dropout as before.  -Link this to a 10-unit output layer as before.  


# TRAINING MODEL FOR CIFAR-10 DATASET
# ------------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
modelc = Sequential()
modelc.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
modelc.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelc.add(MaxPooling2D(pool_size=(2, 2)))
modelc.add(Dropout(0.25))
modelc.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelc.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelc.add(MaxPooling2D(pool_size=(2, 2)))
modelc.add(Dropout(0.25))
modelc.add(Flatten())
modelc.add(Dense(512, activation='relu'))
modelc.add(Dropout(0.5))
modelc.add(Dense(10, activation='softmax'))
modelc.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 32, 32, 32)        896
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 32, 32, 32)        9248
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0
# _________________________________________________________________
# dropout (Dropout)            (None, 16, 16, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 16, 16, 64)        36928
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 8, 8, 64)          0
# _________________________________________________________________
# flatten (Flatten)            (None, 4096)              0
# _________________________________________________________________ 
# dense (Dense)                (None, 512)               2097664
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 512)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                5130
# =================================================================
# Total params: 2,174,362
# Trainable params: 2,174,362

# In compiling the model, we will use a specialised optimizer module: 
# optimizer = keras.optimizers.RMSprop(learning_rate=float(0.0001), weight_decay=1e-6)

from keras import optimizers

optimizer = optimizers.RMSprop(learning_rate=0.0001, weight_decay=1e-6)
modelc.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model using 25 epochs, a batch size of 64 and 20% of the training data for 
# validation. Store the training history in a variable called 'history'.
historyc = modelc.fit(x_trainc, y_trainc, batch_size=64, epochs=25, validation_split=0.2)

# Evaluate the model on the test data, storing the results in a variable called
# 'test_loss' and 'test_acc'.
test_lossc, test_accc = modelc.evaluate(x_testc, y_testc)
print('Test loss:', test_lossc)
print('Test accuracy:', test_accc)
# Test loss: 0.7810584306716919
# Test accuracy: 0.7303000092506409

# Save model to file called 'cifar10_model.h5'.


# # Plot the training and validation accuracy and loss over the epochs, as in Learning-
# # based Exercise One.
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historyc.history['accuracy'], label='Training Accuracy')
plt.plot(historyc.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(historyc.history['loss'], label='Training Loss')
plt.plot(historyc.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Repeat the above steps for the MNIST dataset, using a model with the same structure
# as above, but with input_shape = (28, 28, 1) in the first convolutional layer.

# TRAINING MODEL FOR MNIST DATASET
# ------------------------------------------------------------------------------------------------

modelm = Sequential()
modelm.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
modelm.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelm.add(MaxPooling2D(pool_size=(2, 2)))
modelm.add(Dropout(0.25))
modelm.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelm.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
modelm.add(MaxPooling2D(pool_size=(2, 2)))
modelm.add(Dropout(0.25))
modelm.add(Flatten())   
modelm.add(Dense(512, activation='relu'))
modelm.add(Dropout(0.5))
modelm.add(Dense(10, activation='softmax'))
modelm.summary()
modelm.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

historym = modelm.fit(x_trainm, y_trainm, batch_size=64, epochs=25, validation_split=0.2)

test_lossm, test_accm = modelm.evaluate(x_testm, y_testm) 
print('Test loss:', test_lossm)
print('Test accuracy:', test_accm)
# Test loss: 0.019342005252838135
# Test accuracy: 0.9932000041007996


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historym.history['accuracy'], label='Training Accuracy')
plt.plot(historym.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(historym.history['loss'], label='Training Loss')
plt.plot(historym.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Save to file called 'mnist_model.h5'.
modelm.save('mnist_model.h5')
