import torch
import torch.nn as nn
from Arc_Loss.ArcSoftmax import Arcsoftmax

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.PReLU(),
                                        nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.PReLU(),
                                        nn.MaxPool2d(3, 2))
        self.feature_layer = nn.Sequential(nn.Linear(11 * 11 * 64, 256), nn.BatchNorm1d(256), nn.PReLU(),
                                           nn.Linear(256, 128), nn.BatchNorm1d(128), nn.PReLU(),
                                           nn.Linear(128, 2), nn.PReLU())
        self.arcsoftmax = Arcsoftmax(2, 9)

    def forward(self, x, s, m):
        conv = self.conv_layer(x)
        conv = conv.reshape(x.size(0), -1)
        feature = self.feature_layer(conv)
        out = self.arcsoftmax(feature, s, m)
        out = torch.log(out)
        return feature, out