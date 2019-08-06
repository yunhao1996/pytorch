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
   
   device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
```
`torchvision.datasets.ImageFloder`:数据加载器，也可以通过字面意思简单的理解，处理加载图像的工具。这里面非常重要的参数就是`transform`,输入为图片，输出为处理后的数据。  `transforms.Normalize`:用于数据归一化处理，将前面0-1之间的像素值，再映射到-1 -> 1.   (0.1307, 0.1307, 0.1307)表示RGB每条通道的标准差std，(0.3081, 0.3081, 0.3081)表示RGB每条通道的均值mean。所有的值应该是0.5。这里的数值选择主要是根据网上的经验引用的，我自己的看法是，相比*celebA*数据集，*MNIST*数据集得到的图片像素范围过于单一，映射范围围绕0即可。一系列数据的映射范围的改变，只有一个目的，让模型收敛更稳定，避免过拟合。

在*pytorch*的框架结构中，`torch.utils.data.DataLoader`基本都要用到，负责将数据进行随机的批处理， [为什么要batch_size？](https://blog.csdn.net/qq_42380515/article/details/87885996) 请点击连接。此外，我们还需要打开`shuffle`,随机构建batch.每个`epock`都要对数据进行重新的`shuffle`,目的就是让内存效率和内存容量达到平衡。

`torch.device`:



















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

# 设置随机数种子，重现结果
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 定义参数值
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

if __name__ == '__main__':

    # create the dataset
    dataset = dset.ImageFolder(root=dataroot,  # datatoot
                               transform=transforms.Compose([  # 将多了转换函数组合起来使用
                                   transforms.Resize(image_size),  # 图像扩大为64×64，默认双线性插值
                                   transforms.ToTensor(),  # 将img转化为tensor张量,由（H,W,C）->(C,H,W),顺便压缩到0-1
                                   # 前面的数据是3通道的均值，后面是标准差
                                   transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),  # 数据归一化，转化为-1->1之间，先减均值，在除以标准差
                               ]))
    # print(dataset[0])
    # plt.imshow(dataset[0][0][0],cmap='gray')
    # plt.show()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  # 设置批大小
                                             shuffle=True, num_workers=workers)  # 打乱顺序，设置同时工作的子进程
    # 选择运行设备GPU
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 查看8乘8的图像
    # real_batch1 = next(iter(dataloader))  # iter:用来生成迭代器，遍历数据；next:返回迭代器对像
    # plt.figure(figsize=(8, 8))
    # plt.axis("on")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(real_batch1[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()

    # pytorch初始化网络权重的一种方法,其实也就是初始化可学习的参数
    def weights_init(m):
        # __name__ 表示当前程序运行在哪个模块中
        classname = m.__class__.__name__  # 返回m的名字
        # 如果if满足条件，不会运行elif.
        if classname.find('Conv') != -1:  # 查找classname是否含有字符‘Conv’,有，返回0；没有，返回-1。所以和-1进行比较
           # 将m.weight.data初始化为均值为0，方差为0.02均匀分布中的随机变量,应用于卷积
           nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:  # 应用在网络层批量归一化中
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)  # 将m.bias.data初始化为0


    # 生成器网络


    class Generator(nn.Module):  # 创建Generator子类，括号内指定父类的名称
        def __init__(self, ngpu):  # 初始化父类的属性
            super(Generator, self).__init__()  # 将父类和子类关联，调用父类nn.Moudle的方法__init__(),让Generator实例包含父类的所有属性
            self.ngpu = ngpu
            self.main = nn.Sequential(  # 按照顺序构造神经层，序列容器
                # 输入的是z，100×1×1
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # 转置卷积，输出
                nn.BatchNorm2d(ngf * 8),  # 对每个特征图上的点，进行减均值除方差的操作，affine设置为true(默认)，引入权重w和b两个可学习的参数
                nn.ReLU(inplace=True),  # 括号加不加True计算结果不会有错误，通过对原变量覆盖的方式，释放内存，加速计算
                # 512 x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 翻倍
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. 256 x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. 128 x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. 64 x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. 3 x 64 x 64
            )

        def forward(self, input):
            return self.main(input)


    # 生成器类实例化，self指向类实例本身
    netG = Generator(ngpu).to(device)  # 将实例放到GPU中
    # print(netG)

    # 多GPU运行，数据并行，device.type： 获取设备的类型
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上
    # to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    # print(netG)


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is 3 x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),  # 加快梯度传播，有助于训练
                # state size. 64 x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 128 x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 256 x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 512 x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                # state size. 1 x 1 x 1
            )

        def forward(self, input):
            return self.main(input)


    # 创建判决器实例
    netD = Discriminator(ngpu).to(device)

    # 针对于多个GPU,数据并行
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # 应用权重初始化函数随机初始化所有权重
    #  均值为0，标准差为0.2
    netD.apply(weights_init)

    # Print the model
    # print(netD)

    # 初始化损失函数BCELoss:二分类交叉熵
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)  # 生成size为n * m * e * v的随机数张量

    # 在训练过程中规定真样本和假样本的label
    real_label = 1
    fake_label = 0

    # G 和 D 的优化器均选择Adam
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    # 对于每一个epock
    for epoch in range(num_epochs):
        # 对于dataloader中的每一个batch
        for i, data in enumerate(dataloader, start=0):  # 下标0为标签起始位置

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # 用所有真实的批量样本-batch 训练
            netD.zero_grad()  # 梯度计算为累加方式，所以梯度要更新为0
            # 就是将数据放在device上面，因为初始化的时候没有指定将数据放在哪个device上面所以训练的时候要转化一下
            real_cpu = data[0].to(device)  # 将模型放到显存上
            # 第0维是数据批次的size
            b_size = real_cpu.size(0)
            # 就是给产生一个size大小的值为fill_value的张量，这个张量是被用来当做真实标签的值都是1
            label = torch.full((b_size,), real_label, device=device)
            # D:前向传播
            output = netD(real_cpu).view(-1)
            # 就算真实样本的损失
            errD_real = criterion(output, label)
            # 进行一次反向传播更新梯度
            errD_real.backward()
            # 是对一个tensor去均值并将这个tensor转换成python中的数值
            D_x = output.mean().item()

            ## 训练所有的假样本批量
            # 生成隐向量批量
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # G网络生成假的图像
            fake = netG(noise)
            # 给张量label赋予值fake_label
            label.fill_(fake_label)
            # D网络判别假的图像
            # tensor.detach:将一个张量从graph中剥离出来，不用计算梯度
            # .view(-1)表示reshape成一维度向量，这个一维向量的长度由推断得出
            output = netD(fake.detach()).view(-1)
            # D网络计算所有的假样本批量损失，
            errD_fake = criterion(output, label)
            # 利用反向传播算法计算梯度
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # 获取D网络的loss
            errD = errD_real + errD_fake
            # D网络参数更新
            optimizerD.step()

            #####################################################################
            # (2) Update G network: maximize log(D(G(z))) or minmize -log(D(G(z)))
            #####################################################################
            netG.zero_grad()  # 将模型的参数梯度设为0，用于反向传播
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # 计算G网络的损失
            errG = criterion(output, label)
            # 反向传播计算G网络的梯度
            errG.backward()
            D_G_z2 = output.mean().item()
            # 更新G网络的参数
            optimizerG.step()

            # 每隔50输出损失
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():  # 确定上下文管理器里面的tensor都是不需要计算梯度的，可以减少计算单元的浪费
                    fake = netG(fixed_noise).detach().cpu()  # tensor.cpu():将数据从显存复制到内存里面
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

            # if i % 450 == 0:
            #     # Grab a batch of real images from the dataloader
            #     real_batch = next(iter(dataloader))
            #
            #     # Plot the real images
            #     plt.figure(figsize=(15, 15))
            #     plt.subplot(1, 2, 1)
            #     plt.axis("off")
            #     plt.title("Real Images")
            #     plt.imshow(
            #         np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
            #                      (1, 2, 0)))
            #
            #     # Plot the fake images from the last epoch
            #     plt.subplot(1, 2, 2)
            #     plt.axis("off")
            #     plt.title("Fake Images")
            #     plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            #     plt.show()


```



















