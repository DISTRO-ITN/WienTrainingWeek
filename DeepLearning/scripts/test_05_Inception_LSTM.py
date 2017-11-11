
# coding: utf-8

# In[5]:

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.applications import InceptionV3, VGG19
from keras.layers import TimeDistributed

import numpy as np


# <img src="https://github.com/imatge-upc/activitynet-2016-cvprw/blob/master/misc/images/network_pipeline.jpg?raw=true" height="424" width="400">

# In[ ]:

# # define CNN model
#cnn = Sequential()
#cnn.add(Conv2D(...))
#cnn.add(MaxPooling2D(...))
#cnn.add(Flatten())
# #define LSTM model
#model = Sequential()
#model.add(TimeDistributed(cnn, ...))
#model.add(LSTM(..))
#model.add(Dense(...))


# In[ ]:

#model = Sequential()
# # define CNN model
#model.add(TimeDistributed(Conv2D(...))
#model.add(TimeDistributed(MaxPooling2D(...)))
#model.add(TimeDistributed(Flatten()))
# # define LSTM model
#model.add(LSTM(...))
#model.add(Dense(...))


# In[9]:

K.set_learning_phase(1)
## Define vision model
## Inception (currently doesn't work)
cnn = InceptionV3(weights='imagenet',
                  include_top='False',
                  pooling='avg')

# Works
#cnn = VGG19(weights='imagenet',
#            include_top='False', pooling='avg')

cnn.trainable = False

H=W=229
C = 3
video_input = Input(shape=(None,H,W,C), name='video_input')

encoded_frame_sequence = TimeDistributed(cnn)(video_input)

encoded_video = LSTM(256)(encoded_frame_sequence)

output = Dense(256, activation='relu')(encoded_video)

video_model = Model(inputs=[video_input], outputs=output)

print(video_model.summary())

video_model.compile(optimizer='adam', loss='mean_squared_error')

#features = np.empty((0,1000))

n_samples = 3
n_frames = 50

frame_sequence = np.random.randint(0.0,255.0,size=(n_samples, n_frames, H,W,C))

y = np.random.random(size=(3,256,))
y = np.reshape(y,(-1,256))

print(frame_sequence.shape)

video_model.fit(frame_sequence, y, validation_split=0.0,shuffle=False, batch_size=1, epochs=10)


# In[10]:

x = np.random.randint(0.0,255.0,size=(1, n_frames, H,W,C))
result = video_model.predict(x, batch_size=32, verbose=0)
print(result)


# In[ ]:



