# 程序来源：[DCGAN Tutorial](https://www.cnblogs.com/IvyWong/p/9203981.html)
## 本文主要介绍了该程序的一些个人理解

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

### 3.数据集处理
由于电脑太过垃圾，我选择了把*celebA*数据集换成了*MNIST*,为了偷懒，不过多改参数，我的处理方法是将*MNIST*数据输出为图片进行保存。
#### a.将MNIST转化为图片
```python


```
