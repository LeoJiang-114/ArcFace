import torch
import torch.nn as nn
import torch.nn.functional as F

class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num))).cuda()

    def forward(self, x, s, m):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)
        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)
        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True)
            - torch.exp(s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))
        return arcsoftmax