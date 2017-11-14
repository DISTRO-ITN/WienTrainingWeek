#import needed modules
import sklearn
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123) #for reproductibility

#check that the available versions of the modules are modern enough
from distutils.version import StrictVersion
assert(StrictVersion(sklearn.__version__) >= StrictVersion('0.18.1')) #if error: pip install scikit-learn matplotlib
assert(StrictVersion(tf.__version__) >= StrictVersion('1.0.0')) #if error: pip install tensorflow
assert(StrictVersion(keras.__version__) >= StrictVersion('1.2.2')) #if erro: pip install keras

#import specific components from Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K

# Load pre-shuffled MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#dimensions of the training and test datasets
print(x_train.shape[0], 'training samples. x, y  = ', x_train.shape, y_train.shape)
print(x_test.shape[0], 'test samples. x, y = ', x_test.shape, y_test.shape)

#hyperparameters
batch_size = 128
num_classes = 10
epochs = 12

#image dimensions
img_rows, img_cols = 28, 28

#just formatting for the image arrays. Keras expects images as 3D arrays, even if they are grayscale.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#normalizing image data to values between [0, 1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)

# convert digits in category vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Convolution2D(64, (3, 3), activation='relu')) #conv layer with 64 filters
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten()) #flattens input to connect it to the next layer
model.add(Dense(128, activation='relu')) #fully connected NN with 128 neurons
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) #softmax activation converts scores to probabilities

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#12 epoch cpu-only training should take around ~40 minutes runnning on jupyter
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
