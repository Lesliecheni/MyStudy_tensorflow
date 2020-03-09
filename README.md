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

#### **day1**: 读取fashion_mnist的数据
* 首先介绍一下Fashion MNIST数据集,它是7万张灰度图像组成,可以分成10个类别.每个灰度图像都是28*28像素的图像.我们将使用其中的6万张进行训练网络,另外的1万张来评估准确率.
* 他是为了后面的利用tensorflow进行该数据集的基本分类问题来做准备
