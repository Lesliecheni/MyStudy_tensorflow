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


# 切分训练集，测试集
from sklearn.model_selection import train_test_split

x_train_all ,x_test ,y_train_all, y_test = train_test_split(housing.data, housing.target, random_state = 7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state = 11)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


# In[4]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_valid_scaler = scaler.fit_transform(x_valid)
x_test_scaler = scaler.fit_transform(x_test)


# In[12]:


#多输入
input_wide = keras.layers.Input(shape = [5])
input_deep = keras.layers.Input(shape = [6])
hidden1 = keras.layers.Dense(30, activation= 'relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation= 'relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs = [input_wide, input_deep], outputs = [output])


model.summary()
model.compile(loss="mean_squared_error", optimizer="adam")
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=0.000001)]


# In[21]:


x_train_scaler_wide = x_train_scaler[:, :5]
x_train_scaler_deep = x_train_scaler[:, 2:]
x_valid_scaler_wide = x_valid_scaler[:, :5]
x_valid_scaler_deep = x_valid_scaler[:, 2:]
x_test_scaler_wide = x_test_scaler[:, :5]
x_test_scaler_deep = x_test_scaler[:, 2:]


history = model.fit([x_train_scaler_wide, x_train_scaler_deep], y_train,
                    validation_data = ([x_valid_scaler_wide,x_valid_scaler_deep],y_valid),
                    epochs = 100,
                    callbacks = callbacks       
                   )


# In[ ]:


print(history.history)


# In[22]:


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize = (8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    
plot_learning_curves(history)


# In[23]:


model.evaluate([x_test_scaler_wide, x_test_scaler_deep], y_test)


# In[ ]:




