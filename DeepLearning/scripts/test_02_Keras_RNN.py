# In[2]:

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[7]:

leng=4

data = [[i+j for j in range(leng)] for i in range(100)]
data = np.array(data, dtype=np.float32)
target = [[i+j+1 for j in range(leng)] for i in range(100)]
target = np.array(target, dtype=np.float32)
print(data[:10])
print(target[:10])
data = data.reshape(100, 1, leng)/200
target = target.reshape(100,1,leng)/200


# In[ ]:


model = Sequential()  
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))
model.add(LSTM(leng, input_shape=(1, leng),return_sequences=True,activation='sigmoid'))


# In[ ]:

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(data, target, epochs=10000, batch_size=50,validation_data=(data,target))


# In[17]:

predict = model.predict(data)


# In[18]:

print(predict)


# In[ ]:



