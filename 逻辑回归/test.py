import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as utils
import csv
import torch.nn.functional as F


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
    test_data = pd.read_csv('./new_data/dev1.csv')
    test_features = test_data.iloc[:, 1:9]     
    # print(test_features.head(5))
    #数据预处理：
    #选泽合适索引：
    numeric_features=test_features.dtypes[test_features.dtypes !='object'].index   

    #标准化：
    # test_features[numeric_features]=test_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
    test_features=torch.tensor(test_features[:].values,dtype=torch.float)               # 特征列 
    test_labels=torch.tensor(test_data.Group_1.values,dtype=torch.float).view(-1,1)     # 标签列

    #测试数据
    dataset=utils.TensorDataset(test_features,test_labels)
    test_iter=utils.DataLoader(dataset, 1, shuffle=False)

  
    net=DNN(test_features.shape[1])
    net.load_state_dict(torch.load('./model.pth', map_location = 'cpu'))
    net.eval()


    test_acc, num_correct = 0, 0    # 初始化准确率
    test_prob = []
    total_pred = []
    for i, (x, y) in enumerate(test_iter):
        #正向传播：
        # print(i)
        output=net(x)
        output = F.softmax(output)
        abc, pred_label = output.max(1) 
        # print(output.squeeze()[1])
        num_correct += (pred_label == y.long().squeeze()).sum().item()
        test_acc += get_acc(output, y.long().squeeze())
        total_pred.append(pred_label.item())
        test_prob.append(output.squeeze()[1].item())
    print('Test acc:', num_correct / test_features.shape[0])


    with open('v1_1.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['prob']
        csv_writer.writerow(header)
        for i in range(test_features.shape[0]):
            row = [test_prob[i]]
            csv_writer.writerow(row)
    
    print("ending csv....")


