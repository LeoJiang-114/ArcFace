import torch
import torch.nn as nn
from Arc_Loss.ArcSoftmax import Arcsoftmax

class Facenet(nn.Module):
    def __init__(self):
        super(Facenet, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,128,3,1),#198
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2,2),#99
            nn.Conv2d(128,256,3,1),#97
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2,2),#48
            nn.Conv2d(256,256,3,1),#46
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2,2),#23
            nn.Conv2d(256,128,3,1),#21
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2,2),#10
        )
        self.layer2=nn.Sequential(
            nn.Linear(128*10*10,1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256,2)
        )
        self.arcsoftmax=Arcsoftmax(2,9)

    def forward(self, x,s,m):
        out=self.layer1(x)
        out=torch.reshape(out,(-1,128*10*10))
        feature=self.layer2(out)
        output=self.arcsoftmax(feature,s,m)
        output=torch.log(output)
        return feature,output

if __name__ == '__main__':
    x=torch.randn(2,3,200,200).cuda()
    net=Facenet().cuda()
    feature,output=net(x,1,1)
    print(feature.shape)
    print(output.shape)