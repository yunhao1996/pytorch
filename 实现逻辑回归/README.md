#  构造神经网络，实现动态显示Regression回归（来源：莫烦Python）
```python
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 激励函数都在这

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 扩维，变为（100，1）
y = x.pow(2) + 0.2*torch.rand(x.size())  # x的平方加上一些随机噪声

# plt.scatter(x.numpy(), y.numpy())  # 注意，画图处理的是np类型
# plt.show()

from torch.autograd import Variable
x, y = Variable(x), Variable(y)

# 定义神经网络
class Net(torch.nn.Module):  # 继承torch的Module
    def __init__(self, n_feature, n_hidden, n_output):
        # 这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        super(Net, self).__init__()  # 继承__init__功能,
        # 定义每层用什么样的形式,hidden是属性
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):  # 这同时也是Module 中的forward
        # 正向传播输入值，神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数，将输出的n_hidden加工
        x = self.predict(x)  # 输出值
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
"""
Net(
 (hideen)：linear(1->10)
 (predict): linear(10->1)
)
"""

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入net的所有参数，学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式(均方差)

plt.ion()  # 开启交互模式，显示动态图

for t in range(200):
    prediction = net(x)  # 喂给net训练数据x, 输出预测值

    loss = loss_func(prediction, y)  # 计算两者的误差

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值施加到net的parameter上

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()  # 清除当前活动的轴（运动的线），其他的不变
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(1)  # 每次变化的间隔

plt.ioff()  # 交互结束，窗口保留最终结果
plt.show()
```
