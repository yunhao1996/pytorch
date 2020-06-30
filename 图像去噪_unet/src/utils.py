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

    return config

    
def pred_image(image):
    
    pred_rgb = image[0, :, :, :].permute(1,2,0).cpu().detach().numpy()
    pred_rgb = (pred_rgb+1)/2 *255    # [-1,1] -> [0,1]
         
    return pred_rgb[:,:,::-1]
    
    
    
    
    