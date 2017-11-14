from distutils.version import StrictVersion

import sklearn
assert(StrictVersion(sklearn.__version__) >= StrictVersion('0.18.1'))
sklearn.__version__

import tensorflow as tf
assert(StrictVersion(tf.__version__) >= StrictVersion('1.0.0'))
tf.__version__

import keras
assert(StrictVersion(keras.__version__) >= StrictVersion('1.2.2'))
keras.__version__

import numpy as np
import os
from frame_utils import *
np.random.seed(11235813)  # Allows reproducibility.
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Lambda, Input, concatenate
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.backend import tf as ktf

data_path = "20170920_2b"
test_path = "20170920_2b_"
example_path = "/20170920_172635_441_0"
display = False
batch_size = 4
dim_x = 256
dim_y = 256
channels = 1
input_frames = 10
output_frames = 1
total_frames = 251
training_size = 4
validation_size = 4
epochs = 1000

class DataGenerator(object):
    def __init__(self, training_split=1.0):
        'Initialization'
        self.batch_size = batch_size
        self.training_split = training_split

    def generate(self, data_path):
        # Infinite loop
        while 1:
            # Compute the list of data files (use os.listdir(string))
			files = #
            # Compute how many batches we'll iterate over each epoch.
			total_batches = #
            # Generate a random permutation using np.random.permutation(int).
			random_list = #
			# Iterate over the batches.
            for batch_id in range(total_batches):
                # Find the video folders for the batch.
				start = #
				end = #
				list_IDs_temp = [files[k] for k in random_list[start:end]]
                # Generate data.
                X, y = self.__data_generation(list_IDs_temp, data_path)

                yield X, y
	# Loads a number video frames specified by input_frames and output_frames.
    def __data_generation(self, list_IDs_temp, data_path):
        # Initialization.
        X = np.empty((batch_size, dim_x, dim_y, input_frames))
        y = np.empty((batch_size, dim_x, dim_y, output_frames))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            video_path = os.path.join(data_path, list_IDs_temp[i])
            upper_limit = len(os.listdir(os.path.join(video_path, "render/")))
            starting_frame_index = np.random.randint(upper_limit - (input_frames + 1))
            x_data, y_data = load_frames(video_path, starting_frame_index, display)
            # Store input.
            X[i, :, :, :] = x_data
            # Store output.
            y[i, :, :, :] = np.expand_dims(y_data, axis=2)
        return X, y

"""
for i, _ in enumerate(DataGenerator().generate(data_path)):
	print("Step: " + str(i))
"""

inputs = Input(shape = (dim_y, dim_x, input_frames))
x = inputs
# Create a model based on the following architecture:
# Convolution: stride=1, output_depth=32
# Convolution: stride=2, output_depth=[32, 32, 32, 64, 64, 64, 128]
# Convolution: stride=1, output_depth=[256, 256]
# Resize-convolution + concatenate-convolution: [64, 64, 64, 32, 32, 32, 32] + [64, 64, 64, 32, 32, 32, 1]
# Notes: use a kernel of size (3, 3) for all convolutions; use padding='same'; use activation='sigmoid' for the last layer.

model = Model(inputs=inputs, outputs=x)
model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adadelta())

model.load_weights("models/epoch_{0}.h5".format(epochs))

result = np.zeros([dim_x, dim_y, total_frames], dtype=np.float32)
result[:, :, 0:input_frames] = load_frames(data_path + example_path, 1)[0]
for i in range(total_frames - input_frames - 1):
	result_frame = model.predict(np.expand_dims(result[:, :, i:i + input_frames], axis=0))
	result[:, :, input_frames + i] = np.reshape(result_frame, [256, 256])

save_frames(test_path + example_path, result[:, :, :])
generate_video(test_path + example_path + '/', test_path + example_path + "/result.mp4")

"""
model.fit_generator(DataGenerator().generate(data_path),
		  steps_per_epoch=training_size // batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=DataGenerator().generate(data_path),
          validation_steps=validation_size // batch_size)

model.save_weights("models/epoch_{0}.h5".format(epochs))
"""
