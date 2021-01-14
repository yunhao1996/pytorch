import torch.nn as nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides = 1):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = strides, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if strides != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1 , stride = strides, padding = 0, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, BasicBlock=BasicBlock):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_x = nn.Sequential(
            BasicBlock(in_channels = 64, out_channels = 64, strides = 1),
            BasicBlock(in_channels = 64, out_channels = 64, strides = 1)
        )
        self.conv2_x = nn.Sequential(
            BasicBlock(in_channels = 64, out_channels = 128, strides = 2),
            BasicBlock(in_channels = 128, out_channels = 128, strides = 1)
        )
        self.conv3_x = nn.Sequential(
            BasicBlock(in_channels = 128, out_channels = 256, strides = 2),
            BasicBlock(in_channels = 256, out_channels = 256, strides = 1)
        )
        self.conv4_x = nn.Sequential(
            BasicBlock(in_channels = 256, out_channels = 512, strides = 2),
            BasicBlock(in_channels = 512, out_channels = 512, strides = 1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        out = self.conv(x)
        out = self.conv1_x(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.avg_pool(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out
