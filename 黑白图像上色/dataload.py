import torch
import torchvision.transforms.functional as F
from src.utils import load_config
import cv2
import numpy as np 
import glob
from imageio import imread
from PIL import Image

class LABDataset(torch.utils.data.Dataset):
    
    def __init__(self,config, path):
        super(LABDataset, self).__init__()
        
        self.data = self.load_flist(path)
        self.input_size = config.INPUT_SIZE
        
        
    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, index):
        img = self.load_item(index)
        
        return img
    
    def load_item(self, index):
        data = imread(self.data[index])/255
        data = np.array(data, dtype="float32")
        data = self.resize(data, self.input_size, self.input_size)
        lab_image = cv2.cvtColor(data, cv2.COLOR_RGB2LAB)
        lab_image[:, :, 0] *= 255 / 100
        lab_image[:, :, 1] += 128
        lab_image[:, :, 2] += 128
        lab_image /= 255
     
        return self.to_tensor(lab_image)
        
    def load_flist(self, flist):
        flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
        flist.sort()
        return flist
    
    def resize(self, img, width, height):
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
        
        return img
    
    def to_tensor(self, img):
        img_t = F.to_tensor(img).float()
        
        return img_t

        

# if __name__ == "__main__":
#     config = load_config()
#     lab_dataset = LABDataset(config)
    
#     lab_loader = torch.utils.data.DataLoader(lab_dataset, batch_size=1,
#                  shuffle=True, num_workers=2)
#     for _, i in enumerate(lab_loader):
     