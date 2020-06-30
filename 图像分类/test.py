import argparse
from src.config import Config 
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.vgg1 import VGG16
from utils.utils import load_config,train_tf,test_tf
import os
from torchvision.datasets import ImageFolder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='model checkpoints path')
    parser.add_argument('--weights', type=str, default='./checkpoints/3/model/12-regular.pth', help='the weights file you want to test')  # 修改点， 填入模型的地址
    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')
    
    # load config file
    config = Config(config_path)
   
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        print('\nGPU IS AVAILABLE')
        torch.backends.cudnn.benchmark = True
    else:
        config.DEVICE = torch.device("cpu")

    net = VGG16().to(config.DEVICE)  

    test_set = ImageFolder(config.TEST_PATH,transform=test_tf)
    test_data=torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    
    pth_path = args.weights
    net.load_state_dict(torch.load(pth_path), config.DEVICE)
#     net.load_state_dict(torch.load(pth_path, map_location = 'cpu'))  # cpu 使用这一句， 注释上一句
    # print(net)
    net.eval()

    correct_total = 0.0
    total = 0
    
    for n_iter, (image, label) in enumerate(test_data):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_data)))
        image = Variable(image).cuda()   # 如果是cpu, 把 .cuda() 去掉
        label = Variable(label).cuda()   # 如果是cpu, 把 .cuda() 去掉
        output = net(image)
        _, pred = output.topk(1, 1, largest=True, sorted=True)
        
        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()
                      
        #compute top1 
        correct_total += correct[:, :1].sum()

    print()
    print("Top 1 err: ", 1 - correct_total / len(test_data.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
