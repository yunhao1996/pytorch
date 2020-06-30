import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from model import UNet
import argparse
from datetime import datetime
from tensorboardX import SummaryWriter
from src.utils import load_config, pred_lab2rgb
import os
import numpy as np
import random
from dataload import LABDataset 
import cv2

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default = 'train', help="train or test")    
    args = parser.parse_args()
   
    config = load_config()
    
    # 使用tensorboard
    time_now = datetime.now().isoformat()
    
    if not os.path.exists(config.RUN_PATH):
        os.mkdir(config.RUN_PATH)
    writer = SummaryWriter(log_dir=
            config.RUN_PATH)
    
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
        
    net = UNet(2).to(config.DEVICE)
    print(list(torchvision.models.resnet18(False).children())[7])
    
    optimizer = optim.Adam(net.parameters(), betas=(0.5, 0.999), lr=config.LR)
    loss = nn.L1Loss()
    
    # 加载数据集
    if args.action == 'train':
    
        train_dataset = LABDataset(config, config.TRAIN_PATH)
        len_train = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        iter_per_epoch = len(train_loader)
        train_(config, train_loader, net, optimizer, loss,len_train, iter_per_epoch, writer)
         
    if args.action == "test":
        
        test_dataset = LABDataset(config, config.TEST_PATH)
        test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        test(config, test_loader, net, loss)
    
    
def train_(config, train_loader, net, optimizer, loss,len_train, iter_per_epoch, writer):

    print("start training.......")

    # loop over the dataset multiple times.
    for epoch in range(config.EPOCH):  
        # the generator and discriminator losses are summed for the entire epoch.
        total_loss = 0.0
        
        for i, lab_images in enumerate(train_loader):
            
            # split the lab color space images into luminescence and chrominance channels.
            l_images = lab_images[:, 0, :, :]
            l_images = l_images[:, np.newaxis, :, :]
            c_images = lab_images[:, 1:, :, :]
            
            # shift the source and target images into the range [-0.5, 0.5].
            mean = torch.Tensor([0.5])
            l_images = l_images - mean.expand_as(l_images)   # 0,1 -> -0.5,0.5
            l_images = 2 * l_images                          # -0.5,0.5 -> -1,1
            
            c_images = c_images - mean.expand_as(c_images)
            c_images = 2 * c_images
            # allocate the images on the default gpu device.
            
            l_images = Variable(l_images.to(config.DEVICE))
            batch_size = l_images.shape[0]
            c_images = Variable(c_images.to(config.DEVICE))
            
            fake_images = net(l_images)
            
            fake_loss = loss(fake_images, c_images)
            optimizer.zero_grad()
            fake_loss.backward()
            optimizer.step()

            total_loss += fake_loss.item()
            
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tloss: {:0.4f}'.format(
                total_loss / (i+1),
                epoch=epoch+1,
                trained_samples=(i+1) * batch_size,
                total_samples=len_train
            ))
            
            # 可视化loss
            n_iter = epoch * iter_per_epoch + i + 1
            writer.add_scalar('loss', (total_loss/(i+1)), n_iter)
            
            # 保存图片
            if i % config.SAVE_ITER == 0:
                save_img_path = os.path.join(config.PATH, 'sample')
                if not os.path.exists(save_img_path):
                    os.mkdir(save_img_path)
                
                pred_rgb = pred_lab2rgb(l_images, fake_images)
                bw_rgb = pred_lab2rgb(l_images, c_images, True)
                gt_rgb = pred_lab2rgb(l_images, c_images)
                cv2.imwrite(save_img_path+ '/' +"{}_pred.jpg".format(n_iter), pred_rgb)
                cv2.imwrite(save_img_path+ '/' +"{}_bw.jpg".format(n_iter), bw_rgb)
                cv2.imwrite(save_img_path+ '/' +"{}_gt.jpg".format(n_iter), gt_rgb)        
        
        # save the generator and discriminator state after each epoch.
        if (epoch+1)% config.SAVE_EPOCH == 0:
            model_path = os.path.join(config.PATH,'model')
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            checkpoint_path = os.path.join(model_path,'{name}.pth')
            torch.save(net.state_dict(), checkpoint_path.format(name="net"))
                        
    writer.close()        
    print('Finished Training')
    
def test(config, test_loader, net, loss):
    
    print("start testing ........")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./checkpoints/1/model/net.pth', help='the weights file you want to test')  # 修改点
    args = parser.parse_args()
    
    save_img_path = os.path.join(config.PATH, 'test_output')
    save_gt_path = os.path.join(save_img_path, 'gt')
    save_pred_path = os.path.join(save_img_path, 'pred')
    save_bw_path = os.path.join(save_img_path, 'bw')
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
    if not os.path.exists(save_gt_path):
        os.mkdir(save_gt_path)
    if not os.path.exists(save_pred_path):
        os.mkdir(save_pred_path)
    if not os.path.exists(save_bw_path):
        os.mkdir(save_bw_path)
    
    pth_path = args.weights
    generator.load_state_dict(torch.load(pth_path), config.DEVICE)
#     generator.eval() 
    
    total_loss = 0.0
    
    for i, lab_images in enumerate(test_loader):
        print(i)
        # split the lab color space images into luminescence and chrominance channels.
        l_images = lab_images[:, 0, :, :]
        l_images = l_images[:, np.newaxis, :, :]
        c_images = lab_images[:, 1:, :, :]

        # shift the source and target images into the range [-0.5, 0.5].
        mean = torch.Tensor([0.5])
        l_images = l_images - mean.expand_as(l_images)   # 0,1 -> -0.5,0.5
        l_images = 2 * l_images                          # -0.5,0.5 -> -1,1

        c_images = c_images - mean.expand_as(c_images)
        c_images = 2 * c_images
        # allocate the images on the default gpu device.

        l_images = Variable(l_images.to(config.DEVICE))
        c_images = Variable(c_images.to(config.DEVICE))

        # fake images are generated by passing them through the generator.
        fake_images = net(l_images)
        fake_loss = loss(fake_images, c_images)
        total_loss += fake_loss.item()
        
        # 保存结果
        pred_rgb = pred_lab2rgb(l_images, fake_images)
        bw_rgb = pred_lab2rgb(l_images, c_images, True)
        gt_rgb = pred_lab2rgb(l_images, c_images)
        cv2.imwrite(save_pred_path+ '/' +"{}_pred.jpg".format(i), pred_rgb)
        cv2.imwrite(save_bw_path+ '/' +"{}_bw.jpg".format(i), bw_rgb)
        cv2.imwrite(save_gt_path+ '/' +"{}_gt.jpg".format(i), gt_rgb)
        
    print("the mae is:", total_loss / (i+1))
    print("ending")
            
if __name__=='__main__':
    
    main()
    
