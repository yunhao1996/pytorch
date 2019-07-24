# pytorch-二分类
```python
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 激励函数都在这

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
# 均值2*n_data，标准差为1的正态分布中随机生成
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )，标签为0
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据，0是按行连接)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 数据集，FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # 标签集，LongTensor = 64-bit integer,标签的默认形式
# print(y.numpy())
# plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=y.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


# 定义神经网络
class Net(torch.nn.Module):  # 继承torch的Module
    def __init__(self, n_feature, n_hidden, n_output):
        # 这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        super(Net, self).__init__()  # 继承__init__功能,显示调用，因为他不会主动调用，构造函数中包含调用父类构造函数
        # 定义每层用什么样的形式,hidden是属性
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出,这是属性
        self.out = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):  # 这同时也是Module 中的forward，这是继承的方法
        # 正向传播输入值，神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数，将输出的n_hidden加工
        x = self.out(x)  # 输出值
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)  # 几个类别，几个output
# print(net)  # net的结构
"""
Net(
 (hideen)：linear(2->10)
 (predict): linear(10->2)
)
"""
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入net的所有参数，学习率
loss_func = torch.nn.CrossEntropyLoss()  # 分类采用交叉熵损失

plt.ion()  # 开启交互模式，显示动态图

for t in range(100):

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    out = net(x)  # 喂给net训练数据x, 输出分析层

    loss = loss_func(out, y)  # 计算两者的误差

    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值施加到net的parameter上

    if t % 2 == 0:  # 每两部出图一下
        # plot and show learning process
        plt.cla()  # 清除当前活动的轴（运动的线），其他的不变
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.numpy().squeeze()
        target_y = y.numpy()
        # lw=linewidths
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=100, lw=0, camp='RdYIGn')

        accuracy = sum(pred_y==target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  # 交互结束，窗口保留最终结果
plt.show()

```
