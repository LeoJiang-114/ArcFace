import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import numpy as np
from Arc_Loss.Arch_Datasets import Datasets
from Arc_Loss.Arch_Net import MyNet
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # datasets = torchvision.datasets.MNIST(
    #     root="Datasets/",
    #     train=True,
    #     transform=torchvision.transforms.ToTensor(),
    #     download=True
    # )
    #
    # train_data = data.DataLoader(dataset=datasets, batch_size=2000, shuffle=True, drop_last=True)

    img_path = r"F:\Datasets_Test\Center_Classify\face_img"
    datasets = Datasets(img_path)
    batch_size=50
    train_data = data.DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True,drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MyNet().to(device)
    net_path = r"F:\jkl\Arc_Loss\Net_Save\Face_Arc.pth"
    img_save = r"F:\Reply\ArcFace_Loss\Face"
    if not os.path.exists(img_save):
        os.makedirs(img_save)
    if os.path.exists(net_path):
        net = torch.load(net_path).to(device)

    optimizer = torch.optim.Adam(net.parameters())
    loss_NLL=nn.NLLLoss()

    turn = 0
    while True:
        turn += 1
        plt.clf()
        for i, (data, labels) in enumerate(train_data):
            data, labels = data.to(device), labels.to(device)
            # print(data.shape,labels)
            feature, classify = net(data,1,1)
            # print(classify,type(classify))
            # label_hot = torch.zeros(labels.size(0), 10).scatter_(1, labels.cpu().view(-1, 1), 1).long().to(device)
            # print(label_hot,type(label_hot))

            loss = loss_NLL(classify,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("轮次：{0} -- {1}  Loss: {2}".format(turn, i, loss.float()))
            if i == 0:
                torch.save(net, net_path)
                # if turn%5 == 0:
            f, c, labels = feature.cpu().data, classify.cpu().data, labels.cpu().data
            idx = torch.argmax(c, dim=1)
            dic = {0: 'aqua', 1: 'blue', 2: 'black', 3: 'red', 4: 'green',
                   5: 'pink', 6: 'purple', 7: 'orange', 8: 'yellow'}  #, 9: 'lime'
            # plt.clf()
            for i in range(10):
                plt.scatter(f[labels == i, 0], f[labels == i, 1], c=dic[i], s=1)
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8'], loc='upper right')
        # plt.show()
        plt.savefig("{0}\{1}.jpg".format(img_save,turn))
        #plt.pause(0.1)
        # plt.ioff()