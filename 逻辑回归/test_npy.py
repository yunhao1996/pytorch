import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as utils
import csv
import torch.nn.functional as F


# class DNN(nn.Module):
#     def __init__(self, feature_num, num_classes=6):
#         super(DNN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(feature_num, 32),
#             nn.ReLU(True),
#             #15
#             nn.Linear(32, num_classes),
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
            nn.Conv1d(16, 16, 3, 1 , 1),
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

    return num_correct / total

if __name__ == '__main__':

    #读取数据：  
    train_data = np.load('./batch10_data.npy')[:1800, :]
    test_data = np.load('./batch10_data.npy')[1800:, :]            # 选择后1800个数据用于测试
    test_labels = np.load('./batch10_label.npy')[1800:, :] - 1     # label  数值显示1-6

    #标准化：
    test_data = (test_data - train_data.mean()) / (train_data.std())       # 这里需要注意：测试集的均值和方差信息应该来自于训练集

    test_features=torch.tensor(test_data, dtype=torch.float)               # 特征列 
    test_labels=torch.tensor(test_labels, dtype=torch.float).view(-1,1)    # 标签列

    #测试数据
    dataset=utils.TensorDataset(test_features,test_labels)
    test_iter=utils.DataLoader(dataset, 1, shuffle=False)

  
    net=DNN(test_features.shape[1])
    net.load_state_dict(torch.load('./model.pth', map_location = 'cpu'))
    net.eval()


    test_acc = 0    # 初始化准确率
    test_prob = []
    total_pred = []
    for i, (x, y) in enumerate(test_iter):
        #正向传播：
        x =x.unsqueeze(1)
        output=net(x)
        output = F.softmax(output)
        abc, pred_label = output.max(1) 
        test_acc += get_acc(output, y.long().squeeze())
        total_pred.append(pred_label.item())
        test_prob.append(output.squeeze()[1].item())
    # print('Test acc:', num_correct / test_features.shape[0])
    print('Test acc:', test_acc / (i+1))


# 将结果保存到 .csv 文件
    # with open('v1_1.csv', mode='w', newline='') as submit_file:
    #     csv_writer = csv.writer(submit_file)
    #     header = ['prob']
    #     csv_writer.writerow(header)
    #     for i in range(test_features.shape[0]):
    #         row = [test_prob[i]]
    #         csv_writer.writerow(row)
    
    # print("ending csv....")


