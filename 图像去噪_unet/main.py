import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
from model import UNet
import argparse
from datetime import datetime
from tensorboardX import SummaryWriter
from src.utils import load_config, pred_image
import os
import numpy as np
import random
from dataset import MyDataset 
import cv2

def main(mode):
    
    config = load_config()
    
    # 使用tensorboard
    time_now = datetime.now().isoformat()
    
    if not os.path.exists(config.RUN_PATH):
        os.mkdir(config.RUN_PATH)
    writer = SummaryWriter(log_dir=os.path.join(
            config.RUN_PATH, time_now))
    
    # 随机数种子
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    # INIT GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        print('\nGPU IS AVAILABLE')
        torch.backends.cudnn.benchmark = True
    else:
        config.DEVICE = torch.device("cpu")
        
    net = UNet(3).to(config.DEVICE)
    
    optimizer = optim.Adam(net.parameters(), betas=(0.5, 0.999), lr=config.LR)
    criterion = nn.MSELoss()
   
    # 加载数据集
    if mode == 1:
    
        train_dataset = MyDataset(config, config.TRAIN_PATH)
        len_train = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        iter_per_epoch = len(train_loader)
        train_(config, train_loader, net, optimizer, criterion,len_train, iter_per_epoch, writer)
         
    if mode == 2:
        
        test_dataset = MyDataset(config, config.TEST_PATH)
        test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        test(config, test_loader, net, criterion)
    
    
def train_(config, train_loader, net, optimizer, criterion,len_train, iter_per_epoch, writer):

    print("start training.......")

    # loop over the dataset multiple times.
    for epoch in range(config.EPOCH):  
        
        total_loss = 0.0
        
        for i, (x,y) in enumerate(train_loader):
#             print(x)
            x = Variable(x.to(config.DEVICE))
            batch_size = x.size(0)
            y = Variable(y.to(config.DEVICE))
            
            pred = net(x)
        
            optimizer.zero_grad()
     
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            # print statistics on pre-defined intervals.
            total_loss += loss.item()
            
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tloss: {:0.4f}'.format(
                total_loss / (i+1),
                epoch=epoch+1,
                trained_samples=(i+1) * batch_size,
                total_samples=len_train
            ))
            
            # 可视化loss
            n_iter = epoch * iter_per_epoch + i + 1
            writer.add_scalar('loss', (total_loss/(i+1)), n_iter)
#             writer.add_scalar('d_loss', (d_running_loss / (i+1)), n_iter)
            
            # 保存图片
            if i % config.SAVE_ITER == 0:
                save_img_path = os.path.join(config.PATH, 'sample')
                if not os.path.exists(save_img_path):
                    os.mkdir(save_img_path)
                
                moire = pred_image(x)
                pred_rgb = pred_image(pred)
                gt = pred_image(y)
                cv2.imwrite(save_img_path+ '/' +"{}_moire.jpg".format(n_iter), moire)
                cv2.imwrite(save_img_path+ '/' +"{}_pred.jpg".format(n_iter), pred_rgb)
                cv2.imwrite(save_img_path+ '/' +"{}_gt.jpg".format(n_iter), gt)        
        
        # save the generator and discriminator state after each epoch.
        if (epoch+1)% config.SAVE_EPOCH == 0:
            model_path = os.path.join(config.PATH,'model')
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            checkpoint_path = os.path.join(model_path,'{name}.pth')
            torch.save(net.state_dict(), checkpoint_path.format(name="net"))
                        
    writer.close()        
    print('Finished Training')
    
def test(config, test_loader, net, criterion):
    
    print("start testing ........")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./checkpoints/1/model/net.pth', help='the weights file you want to test')  # 修改点
    args = parser.parse_args()
    
    save_img_path = os.path.join(config.PATH, 'test_output')
    save_gt_path = os.path.join(save_img_path, 'gt')
    save_pred_path = os.path.join(save_img_path, 'pred')
    save_moire_path = os.path.join(save_img_path, 'moire')
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
    if not os.path.exists(save_gt_path):
        os.mkdir(save_gt_path)
    if not os.path.exists(save_pred_path):
        os.mkdir(save_pred_path)
    if not os.path.exists(save_moire_path):
        os.mkdir(save_moire_path)
    
    pth_path = args.weights
    net.load_state_dict(torch.load(pth_path), config.DEVICE) 
    
    total_loss_test = 0.0
    
    for j, (x,y) in enumerate(test_loader):
        print(j)
        
        x = Variable(x.to(config.DEVICE))
        batch_size = x.size(0)
        y = Variable(y.to(config.DEVICE))

        pred = net(x)

        loss = criterion(pred, y)
        total_loss_test += loss.item()
        
        # 保存结果
        moire = pred_image(x)
        pred_rgb = pred_image(pred)
        gt = pred_image(y)
        cv2.imwrite(save_pred_path+ '/' +"{}.jpg".format(j), pred_rgb)
        cv2.imwrite(save_moire_path+ '/' +"{}.jpg".format(j), moire)
        cv2.imwrite(save_gt_path+ '/' +"{}.jpg".format(j), gt)
        
    print("the mse is:", total_loss_test/(j+1))
    print("ending")
            
if __name__=='__main__':
    main(mode=1)