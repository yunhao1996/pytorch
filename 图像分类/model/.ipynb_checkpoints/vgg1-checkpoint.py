import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, num_classes=43):
        super(VGG16, self).__init__()
        self.features1= nn.Sequential(
            #1
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
              )
        self.features2= nn.Sequential(
            #2
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.features3= nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.features4= nn.Sequential(
            #4
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.features5= nn.Sequential(
            #5
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.features6= nn.Sequential(
            #6
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.features7= nn.Sequential(
            #7
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.features8= nn.Sequential(
            #8
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        self.features9= nn.Sequential(
            #9
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        self.features10= nn.Sequential(
            #10
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.features11= nn.Sequential(
            #11
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        self.features12= nn.Sequential(
            #12
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True))
        self.features13= nn.Sequential(
            #13
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )
        self.fc = nn.Sequential(
            #14
            nn.Linear(25088,4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096,num_classes),
            )
        #self.classifier = nn.Linear(512, 10)
 
    def forward(self, x):
        x = self.features1(x) 
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.features6(x)
        x = self.features7(x)
        x = self.features8(x)
        x = self.features9(x)
        x = self.features10(x)
        x = self.features11(x)
        x = self.features12(x)
        out = self.features13(x)

#         print(out.shape)
        out = out.view(out.size(0), -1)
#         print(out.shape)
        out = self.fc(out)
#        print(out.shape)
        return out
