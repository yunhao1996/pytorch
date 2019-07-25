```python
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)    # 设定随机数种子

# Hyper Parameters
EPOCH = 2           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    # transforms 会把像素值从（0， 255）压缩到（0-1）.灰度图片是一个通道
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成 tensor
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)

# print(train_data.train_data.size())  # 图片数据信息
# print(train_data.train_labels.size())  # 数据标签信息
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')  # 灰白显示
# print('%i'% train_data.train_labels[0])
# plt.show()

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)，shuffle:是否随机打乱顺序,一共1200批
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
# 在第1维增加维度1，维度从0开始数，这里要手动压缩
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)  # 10000
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 =nn.Sequential(  # input shape (1, 28, 28)
        nn.Conv2d(  # 二维卷积
            in_channels=1,  # 输入通道数
            out_channels=16,  # 输出通道数
            kernel_size=5,  # 每个卷积核为5*5
            stride=1,  # 卷积步长
            padding=2,  # 填充值为2，输出图像长宽不变
           ),  # 输出大小（16,28,28)
        nn.ReLU(),  # 卷积完使用激活函数

         # 最大池化层，在2*2的空间里向下采样，输出（16，14，14）
        nn.MaxPool2d(kernel_size=2),  # 这里stride默认值为kernel_size
        )
        self.conv2 = nn.Sequential(  # 在来一层卷积层，输入（16，14，14）
            nn.Conv2d(16, 32, 5, 1, 2),  # 数值按照上面的属性排列
            nn.ReLU(),  # 32*14*14
            nn.MaxPool2d(2),  # 筛选出重要的特征
        )  # 32*7*7
        self.out = nn.Linear(32*7*7, 10)  # 全连接层，输入32*7*7，输出10个数

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 展平多维的卷积图成（batch_size,32*7*7）
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output  # 返回的是每个数值也就是索引值的可能性


cnn = CNN()  # 定义实体
# print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    # enumerate：返回索引序列，step是索引值
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            correct = 0
            test_output = cnn(test_x)  # （[2000，10]）
            # 按照行进行取最大值,[1],返回最大值的每个索引值，也就是说，可能性比较大
            pred_y = torch.max(test_output, 1)[1].numpy()
            correct += sum(pred_y == test_y.numpy())
            accuracy = correct/ 2000
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

# test_output = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')

```
