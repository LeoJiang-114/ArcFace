import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)).cuda())
        self.func = nn.Softmax()

    def forward(self, x, s, m):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)
        # 将cosa变小，防止acosa梯度爆炸
        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)
        # 这里再乘回来
        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        # 这里arcsomax的概率和不为1，这会导致交叉熵损失看起来很大，且最优点损失也很大
        # print(torch.sum(arcsoftmax, dim=1))
        # exit()
        # lmsoftmax = (torch.exp(cosa) - m) / (
        #         torch.sum(torch.exp(cosa) - m, dim=1, keepdim=True) - (torch.exp(cosa) - m) + (torch.exp(cosa) - m))

        return arcsoftmax
        # return lmsoftmax
        # return self.func(torch.matmul(x_norm, w_norm))


class ClsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.PReLU(),
                                        nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.PReLU(),
                                        nn.MaxPool2d(3, 2))
        self.feature_layer = nn.Sequential(nn.Linear(11 * 11 * 64, 256), nn.BatchNorm1d(256), nn.PReLU(),
                                           nn.Linear(256, 128), nn.BatchNorm1d(128), nn.PReLU(),
                                           nn.Linear(128, 2), nn.PReLU())
        self.arcsoftmax = Arcsoftmax(2, 10)
        self.loss_fn = nn.NLLLoss()

    def forward(self, x, s, m):
        conv = self.conv_layer(x)
        conv = conv.reshape(x.size(0), -1)
        feature = self.feature_layer(conv)
        out = self.arcsoftmax(feature, s, m)
        out = torch.log(out)
        return feature, out

    def get_loss(self, out, ys):
        return self.loss_fn(out, ys)
        # print(ys*torch.sum(-out, dim=1))
        # exit()
        # return torch.sum(-ys*out)

if __name__ == '__main__':
    weight_save_path = r"F:\jkl\Arc_Loss\arc_loss.pth"
    dataset = datasets.MNIST(root=r"F:\jkl\Arc_Loss\Datasets/", train=True, transform=transforms.ToTensor(),
                             download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=2000, shuffle=True)
    cls_net = ClsNet()
    cls_net.cuda()
    if os.path.exists(weight_save_path):
        cls_net=torch.load(weight_save_path)
    fig, ax = plt.subplots()
    for epoch in range(100000):
        plt.cla()
        for i, (xs, ys) in enumerate(dataloader):
            xs = xs.cuda()
            ys = ys.cuda()
            coordinate, out = cls_net(xs, 1, 1)
            coordinate = coordinate.cpu().detach().numpy()
            loss = cls_net.get_loss(out, ys)
            optimizer = optim.Adam(cls_net.parameters())
            # print([i for i, c in cls_net.named_parameters()])
            # exit()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: {}/{} | Loss: {} ".format(i,epoch,loss.cpu().detach().item()))
            ys = ys.cpu().numpy()
            # plt.ion()
            for j in np.unique(ys):
                xx = coordinate[ys == j][:, 0]
                yy = coordinate[ys == j][:, 1]
                ax.scatter(xx, yy,s=1)
        fig.savefig(r"F:\Reply\ArcFace_Loss\A\{}.jpg".format(epoch))
        torch.save(cls_net, weight_save_path)
        print("save success")
            # plt.show()
            # plt.pause(0.1)
            # plt.ioff()
