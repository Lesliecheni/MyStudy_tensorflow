这将记录我从零开始学习tensorflow2.x板块
==

### 准备工作：安装tensorflow 
#### tensorflow 分为 cpu 版本 和 gpu版本
这里，我们主要了解cpu版本的tensorflow
选择的安装方法为:  cmd 中 执行 pip install tensorflow
* 遇到的坑：如果电脑中没有Microsoft Visual Studio 2015-2019的话， 安装结束后 ，在import的时候会报错
* 安装成功检测： 在cmd中输入  pip install tensorflow
出现一系列的 Requirement already satisfied:...
随后进入python解释器，试着import tensorflow as tf， 如果没有报错，即安装成功！



下面就进入每天的学习把！ 
（每一天的代码里都会有我自己详细的注释，自己查阅资料的笔记，以及自己的思考等...）
* 为了方便不同平台的使用者，每天的代码都会以.ipynb的格式和.py的格式上传。

#### **day1** : [读取fashion_mnist的数据集](https://github.com/Lesliecheni/MyStudy_tensorflow/tree/master/day_1%E8%AF%BB%E5%8F%96fashion_mnist%E6%95%B0%E6%8D%AE "悬停显示")
* 首先介绍一下Fashion MNIST数据集,它是7万张灰度图像组成,可以分成10个类别.每个灰度图像都是28*28像素的图像.我们将使用其中的6万张进行训练网络,另外的1万张来评估准确率.
* 他是为了后面的利用tensorflow进行该数据集的基本分类问题来做准备


#### **day2** : [对fashion_mnist建模并对其进行图像分类测试](https://github.com/Lesliecheni/MyStudy_tensorflow/tree/master/day_2%E5%88%A9%E7%94%A8fashion_mnist%E6%95%B0%E6%8D%AE%E9%9B%86%E5%BB%BA%E6%A8%A1%E5%B9%B6%E5%AF%B9%E5%85%B6%E8%BF%9B%E8%A1%8C%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB "悬停显示")
* 根据day1的数据，来建立预测模型，完成了第一个模型的建立。 并进行对测试集的准确率测试。 

#### **day3** : [对fashion_mnist数据集进行归一化处理](https://github.com/Lesliecheni/MyStudy_tensorflow/tree/master/day_3fashion_mnist%E6%95%B0%E6%8D%AE%E5%BD%92%E4%B8%80%E5%8C%96%E5%A4%84%E7%90%86 "悬停显示")
* 对fashion_mnist数据集进行归一化处理， 对归一化之后的数据集进行归一化建模， 对比和没有归一化数据建立的模型，  发现准确率有了一定程度的提升。所以，对于数据集进行归一化处理还是相当有必要的。

#### **day4** : [回调函数的使用](https://github.com/Lesliecheni/MyStudy_tensorflow/tree/master/day_4%E5%9B%9E%E8%B0%83%E5%87%BD%E6%95%B0%E7%9A%84%E4%BD%BF%E7%94%A8 "悬停显示")
* 建立之前的模型的时候，调用Keras.callbacksAPI里的方法(ModelCheckpoint, EarlyStopping , TensorBoard)
* 还有其他的方法，可以参考中文文档 https://keras.io/zh/callbacks/#_1
* 对于tensorboard 的调用， 可以见day4 里的三张截图， 打开步骤： 在callbacks所在的根目录下，进行cmd命令的调用，然后 输入tensorboard --logdir callbacks, 调用以后 输入（我这里）localhost:6006

#### **day5** : [实现简单的深度神经网络](https://github.com/Lesliecheni/MyStudy_tensorflow/tree/master/day5_%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C "悬停显示")
* 基于之前的数据集，建立最简单的深度神经网络
* 使用for _ in range(20) 建立20层神经网络， 使用relu 和 softmax 的激活函数
