import torch

# a=torch.Tensor([[1,2],
#                 [3,4],
#                 [4,5]])
# b=torch.randn((2,10))
# c=torch.matmul(a,b)/10
# print(c.shape)
# d=torch.acos(c)
# print(d.shape)

import os
from PIL import Image
#
# img_path=r"F:\Datasets_Test\Center_Classify\face_img"
# for img_name in os.listdir(img_path):
#     if img_name[0]=="9":
#         img_open = Image.open(os.path.join(img_path, img_name))
#         img_open.save(r"F:\Datasets_Test\Center_Classify\新建文件夹\0{}".format(img_name[1:]))

import torch.nn as nn

# nll=nn.NLLLoss()
# cross=nn.CrossEntropyLoss()
# softmax=nn.Softmax(dim=1)
# input=torch.Tensor([[-0.7715, -0.6205,-0.2562]])
# target = torch.tensor([0])
#
# output=cross(input,target)
# print(output)#1.3447
#
# log_softmax=torch.log(softmax(input))
# output=nll(log_softmax,target)
# print(output)#1.3447

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # BN = nn.BatchNorm2d()
        # LN = nn.LayerNorm()
        # IN = nn.InstanceNorm2d()
        # GN = nn.GroupNorm()

        self.layer =nn.Sequential(
            nn.Conv2d(1,1,3,1),
            #nn.LayerNorm(26,26)
        )

    def forward(self, x):
        y=self.layer(x)
        return y

net=Net()
x=torch.Tensor([[[[1,1,1],
                  [1,1,1],
                  [1,1,1]]]])
y=net(x)
print(y.shape,y)