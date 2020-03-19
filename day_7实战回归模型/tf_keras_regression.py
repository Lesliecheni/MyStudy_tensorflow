#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd

#查看tf 和 keras 的版本
print(tf.__version__)
print(tf.keras.__version__)


# In[2]:


from sklearn.datasets import fetch_california_housing
#导入加利福利亚的房价信息

housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)


# In[3]:


import pprint

pprint.pprint(housing.data[0:5])
pprint.pprint(housing.target[0:5])


# In[4]:


# 切分训练集，测试集
from sklearn.model_selection import train_test_split

x_train_all ,x_test ,y_train_all, y_test = train_test_split(housing.data, housing.target, random_state = 7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state = 11)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


# In[5]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_valid_scaler = scaler.fit_transform(x_valid)
x_test_scaler = scaler.fit_transform(x_test)


# In[15]:


model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape = x_train.shape[1:]),
    keras.layers.Dense(1),
])

model.summary()
model.compile(loss="mean_squared_error", optimizer='SGD')
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=0.000001)]


# In[16]:


history = model.fit(x_train_scaler, y_train,
                   validation_data = (x_valid_scaler,y_valid),
                    epochs = 100,
                    callbacks = callbacks             
                   )


# In[ ]:


print(history.history)


# In[13]:


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize = (8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    
plot_learning_curves(history)


# In[14]:


model.evaluate(x_test_scaler, y_test)


# In[ ]:




