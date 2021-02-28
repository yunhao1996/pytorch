import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torch.nn.functional as F

# class DNN(nn.Module):
#     def __init__(self, feature_num, num_classes=6):
#         super(DNN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(feature_num, 32),
#             nn.ReLU(True),
#             #15
#             nn.Linear(32 , num_classes),
#             nn.Sigmoid()
#             )

#     def forward(self, x):
#         out = self.fc(x)
#         return out

class DNN(nn.Module):
    def __init__(self, feature_num, num_classes=6):
        super(DNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 32, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128 , num_classes),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def get_acc(output, label):

    total = output.shape[0]
    _, pred_label = output.max(1) 
    num_correct = (pred_label == label).sum().item()
    # print(num_correct)
    return num_correct / total

if __name__ == '__main__':

    #读取数据：
    train_data = np.load('./batch10_data.npy')[:1800, :]   # shape （3600, 128),选择前1800个数据训练
    test_data = np.load('./batch10_data.npy')[1800:, :]    # 选择后1800个数据用于测试
    # print(train_data.shape)
    train_labels = np.load('./batch10_label.npy')[:1800, :] - 1    # label  数值显示1-6
    test_labels = np.load('./batch10_label.npy')[1800:, :] - 1     # label  数值显示1-6

    # print (train_labels.shape)

    train_data = (train_data - train_data.mean()) / (train_data.std()) 
    test_data = (test_data - train_data.mean()) / (train_data.std())       # 测试集的均值方差信息来自于训练集

    train_features=torch.tensor(train_data, dtype=torch.float)   
    train_labels=torch.tensor(train_labels, dtype=torch.float).view(-1,1)   #标签列
    test_features=torch.tensor(test_data, dtype=torch.float)   
    test_labels=torch.tensor(test_labels, dtype=torch.float).view(-1,1)     #标签列

    # 训练数据
    dataset_train=utils.TensorDataset(train_features,train_labels)
    train_iter=utils.DataLoader(dataset_train, 16, shuffle=True)     

    # 测试数据
    dataset_test=utils.TensorDataset(test_features,test_labels)
    test_iter=utils.DataLoader(dataset_test, 1, shuffle=False)

    # #损失函数
    loss=nn.CrossEntropyLoss()
    # 实例化网络、定义优化器
    net=DNN(train_features.shape[1])
    optimizer=optim.Adam(params=net.parameters(), lr=0.001)

    for epoch in range(100):
        train_acc, train_loss = 0, 0    # 初始化准确率
        net.train()
        for i, (x, y) in enumerate(train_iter):
            #正向传播：
            # print(y.size())
            x = x.unsqueeze(1)
            # print(x.size())
            output=net(x)
            #计算损失：
            l=loss(output, y.long().squeeze())
            train_loss += l.item()
            train_acc += get_acc(output, y.long().squeeze())
            #梯度归零
            optimizer.zero_grad()
            #反向传播
            l.backward()
            #优化参数
            optimizer.step()
        print('Training Epoch: {epoch} \tLoss: {:0.4f}\ttrain_Acc: {:0.4f}\tLR: {:0.6f}'.format(
        train_loss/(i+1),
        train_acc/(i+1),
        optimizer.param_groups[0]['lr'],
        epoch=epoch
    ))
        
    torch.save(net.state_dict(), './model.pth')
