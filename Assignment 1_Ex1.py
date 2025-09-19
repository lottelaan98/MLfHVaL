from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
num_classes = 10 
num_features = 784

x_train = x_train.reshape(-1, num_features)
x_test = x_test.reshape(-1, num_features)
x_test = x_test/255
x_train = x_train/255

y_test = keras.utils.to_categorical(y_test, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)

model = keras.Sequential()
model.add(keras.layers.Dense(256, input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128,
epochs=12, verbose=1, validation_split=0.2)

#plt.subplot(1,2,1)
#plt.plot(history.history['loss'], label='Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
#plt.legend()

#plt.subplot(1,2,2)
#plt.plot(history.history['accuracy'], label='Accuracy')
#plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#plt.legend()

#plt.show()

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('accuracy:', accuracy)
########

model2 = keras.Sequential()
model2.add(keras.layers.Dense(256, input_shape=(784,), activation='relu'))
model2.add(keras.layers.Dense(10, activation='softmax'))
model2.summary()

model2.compile(loss='categorical_crossentropy',
optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

history2 = model2.fit(x_train, y_train, batch_size=128,
epochs=12, verbose=1, validation_split=0.2)

#plt.subplot(1,2,1)
#plt.plot(history2.history['loss'], label='Loss')
#plt.plot(history2.history['val_loss'], label='Validation Loss')
#plt.legend()

#plt.subplot(1,2,2)
#plt.plot(history2.history['accuracy'], label='Accuracy')
#plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
#plt.legend()
#plt.show()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
activation="relu", input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
#model.add(keras.layers.Dropout(rate=0.25)) #q7
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
#model.add(keras.layers.Dropout(rate=0.5)) #q7
model.add(keras.layers.Dense(10, activation="softmax"))
#model.summary()

model.compile(loss='categorical_crossentropy',
optimizer=keras.optimizers.Adadelta(learning_rate=float(1)),
metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128,
epochs=6, verbose=1, validation_split=0.2)

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('accuracy 2:', accuracy)
