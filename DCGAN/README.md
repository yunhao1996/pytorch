# 程序来源：[DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
## 本文主要介绍了该程序的一些个人理解，只讲代码，相关运行结果没有。

### 1.导入工具包
```python
# from __future__ import print_function  # 把下一版本的特性导入当前版本
import argparse  # 命令行解析模块
import os  #
import random  # 设置随机数种子
import torch  # 类似于numpy的通用数组库，比如数据形式的转化
import torch.nn as nn  # 具有共同层和成本函数的神经网络
import torch.nn.parallel  #
import torch.backends.cudnn as cudnn
import torch.optim as optim  # 具有通用优化算法的优化包
import torch.utils.data as DATA  # 数据加载类
import torchvision.datasets as dset  # 该模块下包含MNIST数据集
import torchvision.transforms as transforms  # 图像转化操作类
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
```
### 2.设置随机数种子

```python
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
```
程序中存在许多随机产生

### 3.定义参数
```python
dataroot = "./mnist/raw"  # 数据集的根目录
workers = 2  # DataLoader进行数据预处理及数据加载使用进程数
batch_size = 128  # 一次batch进入模型的图片数目
image_size = 64  # 原始图片重采样进入模型前的大小
nc = 3  # 输入图像中颜色通道的数目。对于彩色图像
nz = 100  # 初始噪音向量的大小
ngf = 64  # 生成网络中基础feature数目
ndf = 64  # 判别网络中基础feature数目
num_epochs = 5  # 训练5次全部样本
lr = 0.0002  # 学习率
beta1 = 0.5  # 使用Adam优化算法中的β1参数值
ngpu = 1  # 可用的GPU数量
```
结合具体的程序理解参数的意思。


### 4.数据集处理
由于电脑太过垃圾，我选择了把*celebA*数据集换成了*MNIST*,为了偷懒，不过多改参数，我的处理方法是将*MNIST*数据输出为图片进行保存。
#### a.将MNIST转化为图片
```python
# 该程序单独放在一个.py文件中
import torchvision  # mnist数据集就在这里
import scipy.misc
import os  # 处理文件和目录的包

DOWNLOAD_MNIST = False  

train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST)  #下载，处理数据集

save_dir = "mnist/ras/"  # 设置图片路径，绝对路径和相对路径都可以
if os.path.exists(save_dir) is False:  # os.path.exit():验证括号内的路径是否存在，返回布尔值
    os.makedirs(save_dir)  # 如果路径存在，创建路径文件。英文， make dirs:创建目录

for i in range(60000):
    image_array, label = train_data[i]  # 获取信息train_data.train_data和train_data.train_label
    image_array = image_array.resize(28, 28)  # 将每个图片的像素大小调整为28 x 28
    filename = save_dir + '%d.jpg' % i  # 设置图片的路径，名字和格式
    print(filename)  # 打印每张图片的详细地址
    print(train_data.train_labels[i])  # 输出每张图片的标签。
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)  # 保存图片  
```
*python*非常方便的一点还包括它的函数名字，有时候通过字面意思，便能够理解该函数的用法。这里讲解一下下载数据集这个程序。  

`torchvision.dataset.MNIST() `:从汉语意思上，*orchvision*数据集下面的*MNIST*。里面的一些常用参数：  

`root='path'`: 设置的下载保存路径  

`train=True`: *MNIST*数据集下面包含两部分，测试集`train_data`和验证集`test_data`,两部分独立存在。该语句设置为`True`，意思为只下载测试集。设置为`False`,只下载验证集。顺便一提，比如测试集`train_data`,下面又有两个属性，包括`train_data.train_data`(图像数据，也就是常用的X)和`train_data.train_label`(图片表示的数字，标签Y)

程序的最后一步是保存图片，我使用的是`scipy.misc.toimage`,这个函数在*scipy*包1.2版本开始，已经取消，这里提出另一种方法：
```python
    scipy.misc.imsave(filename, image_array)
```

#### b.数据集的处理
```python
   dataset = dset.ImageFolder(root=dataroot,  # 数据集的位置
                               transform=transforms.Compose([  # 将多个转换函数组合起来使用
                                   transforms.Resize(image_size),  # 图像扩大为64×64，默认双线性插值
                                   transforms.ToTensor(),  # 将img转化为tensor张量,由（H,W,C）->(C,H,W),像素自动压缩到0-1
                                   transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)), 
                               ]))
   dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  # 设置批大小
                                             shuffle=True, num_workers=workers)  # 打乱顺序，设置同时工作的子进程                            
```
`torchvision.datasets.ImageFloder`:数据加载器，也可以通过字面意思简单的理解，处理加载图像的工具。这里面非常重要的参数就是`transform`,输入为图片，输出为处理后的数据。  `transforms.Normalize`:用于数据归一化处理，将前面0-1之间的像素值，再映射到-1 -> 1.   (0.1307, 0.1307, 0.1307)表示RGB每条通道的标准差std，(0.3081, 0.3081, 0.3081)表示RGB每条通道的均值mean。所有的值应该是0.5。这里的数值选择主要是根据网上的经验引用的，我自己的看法是，相比*celebA*数据集，*MNIST*数据集得到的图片像素范围过于单一，映射范围围绕0即可。一系列数据的映射范围的改变，只有一个目的，让模型收敛更稳定，避免过拟合。

在*pytorch*的框架结构中，`torch.utils.data.DataLoader`基本都要用到，负责将数据进行随机的批处理， [为什么要batch_size？](https://blog.csdn.net/qq_42380515/article/details/87885996)

















