import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils

class DNN(nn.Module):
    def __init__(self, feature_num, num_classes=2):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_num, feature_num - 2),
            nn.ReLU(True),
            #15
            nn.Linear(feature_num -2 , num_classes),
            nn.Sigmoid()
            )

    def forward(self, x):
        out = self.fc(x)
        return out

def get_acc(output, label):

    total = output.shape[0]
    _, pred_label = output.max(1) 
    num_correct = (pred_label == label).sum().item()

    return num_correct / total

if __name__ == '__main__':

    #读取数据：
    train_data = pd.read_csv('./new_data/dev1.csv')
    train_features = train_data.iloc[:,1:9]      # 18 : 26
    print (train_features.head(5))

    # #数据预处理：
    # #选泽合适索引：
    numeric_features=train_features.dtypes[train_features.dtypes !='object'].index   

    # #标准化：
    # train_features[numeric_features]=train_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
    # # print(train_features.head(5))

    # # #通过values得到Numpy的数据，并转换成Tensor
    n_train = train_data.shape[0]       #样本个数

    train_features=torch.tensor(train_features[:].values,dtype=torch.float)   #
    train_labels=torch.tensor(train_data.Group_1.values,dtype=torch.float).view(-1,1)   #标签列
        # #训练数据
    dataset=utils.TensorDataset(train_features,train_labels)
    train_iter=utils.DataLoader(dataset, 16, shuffle=True)

    # #损失函数
    loss=nn.CrossEntropyLoss()
    # 实例化网络、定义优化器
    net=DNN(train_features.shape[1])
    optimizer=optim.Adam(params=net.parameters(), lr=0.002)

    for epoch in range(300):
        train_acc, train_loss = 0, 0    # 初始化准确率
        for i, (x, y) in enumerate(train_iter):
            #正向传播：
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

        print('Training Epoch: {epoch} \tLoss: {:0.4f}\tAcc: {:0.4f}\tLR: {:0.6f}'.format(
                train_loss/(i+1),
                train_acc/(i+1),
                optimizer.param_groups[0]['lr'],
                epoch=epoch
            ))
        
    torch.save(net.state_dict(), './model.pth')
