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
input_frames = 3
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
            # Compute the list of data files.
            files = os.listdir(data_path)
            # Compute how many batches we'll iterate over each epoch.
            total_batches = int(len(files) * self.training_split // batch_size)
            # Used to sample from the training or test data.
            random_list = np.random.permutation(int(len(files) * self.training_split))
            for batch_id in range(total_batches):
                # Find the video folders for the batch.
                list_IDs_temp = [files[k] for k in random_list[batch_id * batch_size:(batch_id + 1) * batch_size]]
                # Generate data.
                X, y = self.__data_generation(list_IDs_temp, data_path)

                yield X, y

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
# Used for skip-connections.
encoder_layers = []
# Encoder.
encoder_n = 8
for i in range(encoder_n):
	strides = 2 if i > 0 else 1
	# 32, 32, 32, 32, 64, 64, 64, 128
	out_channels = 2 ** ((i - 1) // 3 + 5) if i > 0 else 2 ** 5
	x = Convolution2D(out_channels, kernel_size=(3, 3), activation='relu', strides=strides, padding='same')(x)
	encoder_layers.append(x)
    
# Fully connected emulation through convolutions.
flat_n = 2
for i in range(flat_n):
	strides = 1
	# 256, 256
	out_channels = 2 ** 8
	x = Convolution2D(out_channels, kernel_size=(3, 3), activation='relu', strides=strides, padding='same')(x)
        
# Decoder.
decoder_n = 7
for i in range(decoder_n):
	# Resize-convolution.
	strides = 1
	reverse_i = decoder_n - i - 1
	size = [dim_x // 2 ** (decoder_n - i - 1), dim_y // 2 ** (decoder_n - i - 1)]
	# 64, 64, 64, 32, 32, 32, 32
	out_channels = 2 ** ((decoder_n - i - 2) // 3 + 5) if i != decoder_n - 1 else 2 ** 5
	x = Lambda(lambda y: ktf.image.resize_images(y, size))(x)
	x = Convolution2D(out_channels, kernel_size=(3, 3), activation='relu', strides=strides, padding='same')(x)
	# Concatenation.
	x = concatenate([x, encoder_layers[decoder_n - i - 1]], axis=3)
	# 64, 64, 64, 32, 32, 32, 1
	out_channels = 2 ** ((decoder_n - i - 2) // 3 + 5) if i != decoder_n - 1 else channels * output_frames
	activation = 'relu' if i != decoder_n - 1 else 'sigmoid'
	x = Convolution2D(out_channels, kernel_size=(3, 3), activation=activation, strides=strides, padding='same')(x)
        
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
