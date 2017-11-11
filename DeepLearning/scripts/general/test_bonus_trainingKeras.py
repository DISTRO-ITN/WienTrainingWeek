from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from keras import backend as K
K.set_image_dim_ordering('tf')

from test_bonus_importDataKeras import DataGenerator
import os

# path to the model weights files.
#weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'
# dimensions of our images.

training_data_dir = os.path.join('input_keras','POI','training')
validation_data_dir = os.path.join('input_keras','POI','validation')
nb_train_samples = 8000
nb_validation_samples = 2000
epochs = 50
batch_size = 8


# Generators
training_generator = DataGenerator(training_data_dir, batch_size).generate()
validation_generator = DataGenerator(validation_data_dir, batch_size).generate()

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False,  input_shape = (512,512,3)) #tf 224,224,3 #th 3,224,224
print('Model loaded.')


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu', kernel_initializer='truncated_normal'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4, activation='sigmoid', kernel_initializer='truncated_normal'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path) 

# add the model on top of the convolutional base
super_model = Sequential()
super_model.add(model)
super_model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
super_model.compile(loss='mean_squared_error',optimizer="rmsprop")

tensorboard = TensorBoard(log_dir='./logsKeras', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# fine-tune the model
super_model.fit_generator(
    generator=training_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[tensorboard])