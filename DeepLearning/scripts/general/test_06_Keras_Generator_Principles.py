
# coding: utf-8

# # Generator in Keras
# # For Working Example Check in my PC

# # Data Generation

# In[71]:

import numpy as np
import pandas as pd
data = np.random.rand(200,2)
expected = np.random.randint(2, size=200).reshape(-1,1)

dataFrame = pd.DataFrame(data, columns = ['a','b'])
expectedFrame = pd.DataFrame(expected, columns = ['expected'])

dataFrameTrain, dataFrameTest = dataFrame[:100],dataFrame[-100:]
expectedFrameTrain, expectedFrameTest = expectedFrame[:100],expectedFrame[-100:]


# # Generator

# In[72]:

def generator(X_data, y_data, batch_size):

  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch/batch_size
  counter=0

  while 1:

    X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    counter += 1
    yield X_batch,y_batch

    #restart counter to yeild data in the next epoch as well
    if counter <= number_of_batches:
        counter = 0


# # Keras Model

# In[ ]:

# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.utils import np_utils


model = Sequential()
model.add(Dense(12, activation='relu', input_dim=dataFrame.shape[1]))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#Train the model using generator vs using full batch
batch_size = 8

model.fit_generator(generator(dataFrameTrain,expectedFrameTrain,batch_size), epochs=200,steps_per_epoch = dataFrame.shape[0]/batch_size, validation_data=generator(dataFrameTest,expectedFrameTest,batch_size*2),validation_steps=dataFrame.shape[0]/batch_size*2)

#without generator
#model.fit(x = np.array(dataFrame), y = np.array(expected), batch_size = batch_size, epochs = 3)


# # Bonus - Thread Safe Generator example

# In[ ]:

class createBatchGenerator:

    def __init__(self, driving_log,batch_size=32):
        self.driving_log = driving_log
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
           batch_images = np.zeros((batch_size, 66, 200, 3))
           batch_steering = np.zeros(batch_size)

           for i in range(self.batch_size):
               x,y = get_preprocessed_row(self.driving_log)
               batch_images[i]=x
               batch_steering[i]=y
           return batch_images, batch_steering


# # Bonus... another example

# In[ ]:

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 32, dim_y = 32, dim_z = 32, batch_size = 32, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle

  def generate(self, labels, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(labels, list_IDs_temp)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, labels, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
      y = np.empty((self.batch_size), dtype = int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store volume
          X[i, :, :, :, 0] = np.load(ID + '.npy')

          # Store class
          y[i] = labels[ID]

      return X, sparsify(y)

