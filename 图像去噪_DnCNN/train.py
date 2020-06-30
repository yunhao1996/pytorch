import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *
from datetime import datetime
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=108, help="Training batch size")
parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=10, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--log_dir", type=str, default="checkpoints/6/runs", help='path of saving tensorboardx')
parser.add_argument("--save_model", type=str, default="./checkpoints/6/model/noise_15", help='path of saving model')
parser.add_argument("--image_path", type=str, default="./checkpoints/6/sample", help='path of saving images of train')
parser.add_argument("--noiseL", type=float, default=15, help='noise level')
opt = parser.parse_args()

def main():
    # 加载训练集
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    
    # 加载模型
    net = DnCNN(channels=1, num_of_layers=17)
    net.apply(weights_init_kaiming)                         # 权重初始化
    
    # 使用GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
#     criterion.cuda()
    
    # 定义损失和优化器
    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    # 使用tensorboardx可视化训练曲线和指标
    time_now = datetime.now().isoformat()
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    writer = SummaryWriter(log_dir=os.path.join(
            opt.log_dir, time_now))

    step = 0
    for epoch in range(opt.epochs):
        
        # 设置学习率
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
#             current_lr = opt.lr / 10.
            current_lr = opt.lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        
        # 开始训练
        total_loss = 0
        psnr_train = 0
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data

            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            imgn_train = img_train + noise
#             print(imgn_train.shape)
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            
            # 统计loss和计算psnr，并显示
            out_train = torch.clamp(imgn_train-out_train, 0., 1.)
            psnr_train += batch_PSNR(out_train, img_train, 1.)
            total_loss += loss.item()
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), total_loss/(i+1), psnr_train/(i+1)))
            writer.add_scalar('loss', total_loss/(i+1), step)
            writer.add_scalar('PSNR on training data', psnr_train/(i+1), step)
            
            # 保存训练图片和模型
            step += 1
            if step % 500 ==0:
                if not os.path.exists(opt.image_path):
                    os.mkdir(opt.image_path)
                cv2.imwrite(opt.image_path+ '/' +"{}_pred.jpg".format(step), save_image(out_train))
                cv2.imwrite(opt.image_path+ '/' +"{}_input.jpg".format(step), save_image(imgn_train))
                cv2.imwrite(opt.image_path+ '/' +"{}_gt.jpg".format(step), save_image(img_train))
        if not os.path.exists(opt.save_model):
            os.makedirs(opt.save_model)
        torch.save(model.state_dict(), os.path.join(opt.save_model, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)

    main()
