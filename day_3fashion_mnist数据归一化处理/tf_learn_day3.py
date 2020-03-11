#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#查看tf 和 keras 的版本
print(tf.__version__)
print(tf.keras.__version__)


# In[9]:


# 从 keras.datasets 里面 加载 fashion_mnist 的数据集（很多种类的黑白图片）
fashion_mnist = keras.datasets.fashion_mnist

#拆分训练集和测试集
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

#把训练集拆分成 训练集和验证集   验证集取前5000个数据， 训练集取5000以后的数据
x_vaild, x_train = x_train_all[:5000], x_train_all[5000:]
y_vaild, y_train = y_train_all[:5000], y_train_all[5000:]

#打印 训练集、验证集、测试集的格式    （通过结果可以发现，验证集有5000个28*28的numpy数组，像素位在0~255 的数据，训练集有55000个，测试集有10000个）
print(x_vaild.shape, y_vaild.shape)
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)


# In[20]:


print(np.max(x_train), np.min(x_train))


# In[18]:


#进行数据归一化   # x =  (x-u) / std   u:均值   std:方差    --> 使用fit_transform方法实现
import sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)

x_vaild_scaled = scaler.transform(x_vaild.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)

x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)


# In[22]:


print(np.max(x_train_scaled), np.min(x_train_scaled))


# In[24]:


#tf.keras.models.Sequential()

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

#relu : y = max(0, x)
#softmax: 将向量变成概率分布   x = [x1, x2, x3] ,
#                          y = [e^x1/sum, e^x2/sum, e^x3/sum], 注意 此处的x1，x2, x3为x的一次方，二次方，三次方
#                          sum = e^x1 + e^x2 + e^x3 , 注意 此处的x1，x2, x3为x的一次方，二次方，三次方

#why sparce: y->index.  y->one_hot->[向量] , 如果y已经是向量即categorical_crossentropy, 如果不是即sparse_...
model.compile(loss = "sparse_categorical_crossentropy",  
               optimizer = "adam", #优化器
               metrics = ["accuracy"])


# In[33]:


#使用fit方法进行训练模型，  epoches为训练次数， 结果中loss为待减少的损失值， accuracy为准确率, 验证准确率使用的数据集为x_vaild
history = model.fit(x_train_scaled, y_train, epochs = 10,
         validation_data = (x_vaild_scaled, y_vaild))  


# In[ ]:


# 未进行归一化方法的数据训练后模型的vaild_acc 达到 0.7626
# 进行归一化方法后的数据训练后模型的vaild_acc达到 0.8954


# In[34]:


#使用model.evaluate() 方法来验证测试集的准确率
test_loss , test_acc = model.evaluate(x_test_scaled, y_test) 
print("   test_acc:",test_acc)


# In[ ]:


# 未归一化   ：test_acc 0.7498
# 归一化之后 ：test_acc 0.8792


# In[ ]:


# 通过对比前后归一化数据 模型的准确率变化， 可以看出 对数据进行归一化处理还是非常有必要的

