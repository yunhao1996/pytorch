import argparse
from .config import Config
import os
import torch
import cv2

def load_config(mode=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints/1', help='model checkpoints path (default: ./checkpoints)')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1

    # test mode
    elif mode == 2:
        config.MODE = 2
        # config.INPUT_SIZE = 256
        # config.VAL_FLIST = args.input
        # config.RESULTS = args.output

    return config

    
def pred_lab2rgb(l_img, c_img, is_bw=False):
    
    pred_lab = torch.cat([l_img, c_img], 1)
    pred_lab = pred_lab[0, :, :, :].permute(1,2,0).cpu().detach().numpy()
    pred_lab = (pred_lab+1)/2 *255    # [-1,1] -> [0,1]
    pred_lab[:, :, 2]  -= 128
    pred_lab[:, :, 1]  -=128
    pred_lab[:, :, 0]  *=100/255
    pred_rgb = (cv2.cvtColor(pred_lab, cv2.COLOR_LAB2RGB) * 255)[:,:,::-1]
    
    if is_bw:
        pred_rgb = (cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2GRAY))
        
    return pred_rgb
    
    
    
    
    