import torch
from torch import optim
from torch import nn
import argparse
from model.vgg1 import VGG16
# from model.resnet import
import os
import numpy as np
import random
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import _LRScheduler
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from utils.utils import WarmUpLR,get_acc,load_config,train_tf,test_tf
from datetime import datetime

def main(mode=None):
    
    # 加载超参数
    config = load_config(mode)
    
    # 随机数种子
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    # 记载训练集和测试集
    train_set = ImageFolder(config.TRAIN_PATH, transform=train_tf) 
    length_train = len(train_set)
    train_data=torch.utils.data.DataLoader(train_set,batch_size=config.BATCH_SIZE,shuffle=True)
    iter_per_epoch = len(train_data)

    test_set = ImageFolder(config.TEST_PATH, transform=test_tf)
    length_test = len(test_set)
    test_data=torch.utils.data.DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 选择GPU 或 CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        print('\nGPU IS AVAILABLE')
        torch.backends.cudnn.benchmark = True
    else:
        config.DEVICE = torch.device("cpu")

    # 网络结构
    net = VGG16().to(config.DEVICE)
    print('The Model is VGG16\n')  
    
    # 使用 tensorboardx
    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    writer = SummaryWriter(log_dir = config.LOG_DIR)

    # optimizer and loss function
    optimizer = optim.SGD(net.parameters(),lr=config.LR, momentum=0.9,weight_decay=5e-4)
    loss_function = nn.CrossEntropyLoss()

    # warmup
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.MILESTONES,gamma=1)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * config.WARM)
                 
    # create checkpoint folder to save model
    model_path = os.path.join(config.PATH,'model')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    checkpoint_path = os.path.join(model_path,'{epoch}-{type}.pth')
                 
    best_acc = 0.0
    a = config.EPOCH

    for epoch in range(1, config.EPOCH):

        if epoch > config.WARM:
            train_scheduler.step(epoch)
    
        ### train ###
        net.train()   
        train_loss = 0.0 
        train_correct = 0.0

        for i, data in enumerate(train_data):

            if epoch <= config.WARM:
                warmup_scheduler.step()

            length = len(train_data)
            image, label = data
            image, label = image.to(config.DEVICE),label.to(config.DEVICE)

            output = net(image)
            train_correct += get_acc(output, label)
            loss = loss_function(output, label)
            train_loss +=loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            last_layer = list(net.children())[-1]
            n_iter = (epoch-1) * iter_per_epoch +i +1
            
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAcc: {:0.4f}\tLR: {:0.6f}'.format(
                train_loss/(i+1),
                train_correct/(i+1),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=i * config.BATCH_SIZE + len(image),
                total_samples=length_train
            ))
            writer.add_scalar('Train/lr',optimizer.param_groups[0]['lr'] , n_iter)
            writer.add_scalar('Train/loss', (train_loss/(i+1)), n_iter)
            writer.add_scalar('Train/acc', (train_correct/(i+1)), n_iter)
        
        ## eval ### 
        ## 原本是训练完一次，评估一次，为了节约时间，我给注释掉了##
        if epoch%1==0:
#             net.eval()
#             test_loss = 0.0    
#             test_correct = 0.0

#             for i, data in enumerate(test_data):
#                 images, labels = data
#                 images, labels = images.to(config.DEVICE),labels.to(config.DEVICE)

#                 outputs = net(images)
#                 loss = loss_function(outputs, labels)
#                 test_loss += loss.item()
#                 test_correct += get_acc(outputs, labels)

#                 print('Testing: [{test_samples}/{total_samples}]\tAverage loss: {:.4f}, Accuracy: {:.4f}'.format(
#                 test_loss /(i+1),
#                 test_correct / (i+1),
#                 test_samples=i * config.BATCH_SIZE + len(images),
#                 total_samples=length_test))

#             writer.add_scalar('Test/Average loss', (test_loss/(i+1)), n_iter)
#             writer.add_scalar('Test/Accuracy', (test_correct/(i+1)), n_iter)
#             print()

            #start to save best performance model 
#             acc = test_correct/(i+1)  
#             if epoch > config.MILESTONES[1] and best_acc < acc:
#                 torch.save(net.state_dict(), checkpoint_path.format(epoch=epoch, type='best'))
#                 best_acc = acc
#                 continue

            # 保存模型
            if not epoch % config.SAVE_EPOCH:
                torch.save(net.state_dict(), checkpoint_path.format(epoch=epoch, type='regular'))
    writer.close()
    
if __name__ == "__main__":
    
    main()
    
