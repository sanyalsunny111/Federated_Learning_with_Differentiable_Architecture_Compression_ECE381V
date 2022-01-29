import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, inter_planes, out_planes, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=stride, padding=1, groups=inter_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        self.inter_planes = inter_planes
        self.out_planes = out_planes
    def forward(self, x):
        a = F.relu(self.bn1(self.conv1(x)))
        a = F.relu(self.bn2(self.conv2(a)))
        a = self.bn3(self.conv3(a))
        if self.stride==1 :
            b = self.shortcut(x)
            a=a+b
            return a
        else:
            return a 


class mbn2_nas(nn.Module):
    # (inter_planes, out_planes, stride)
    cfg = [(1*16,  16,  1),
           (6*24,  24,  1), # NOTE: change stride 2 -> 1 for CIFAR10
           (6*24,  24,  1),
           (6*32,  32,  2),
           (6*32,  32,  1),
           (6*32,  32,  1),
           (6*64,  64,  2),
           (6*64,  64,  1),
           (6*64,  64,  1),
           (6*64,  64,  1),
           (6*96,  96,  1),
           (6*96,  96,  1),
           (6*96,  96,  1),
           (6*160, 160, 2),
           (6*160, 160, 1),
           (6*160, 160, 1),
           (6*320, 320, 1)]

    def __init__(self,cfg, num_classes=10):
        super(mbn2_nas, self).__init__()
        self.cfg=cfg

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for inter_planes, out_planes, stride in self.cfg:
            layers.append(Block(in_planes, inter_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        a = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(a)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def maximal_fm(self):
        input_case = torch.randn(1, 3, 32, 32)
        a=self.conv1(input_case)
        a_size=a.size()
        max_fm_size = 1
        for i in a_size:
            max_fm_size = max_fm_size*i 
        for i, block in enumerate(self.layers):
            a=block(a)
            a_size=a.size()
            # print(a_size)
            tmp_max_fm_size = max(block.inter_planes, block.out_planes)*a_size[2]*a_size[3]
            if tmp_max_fm_size>max_fm_size:
                max_fm_size = tmp_max_fm_size

        a=self.conv2(a)
        a_size=a.size() 
        tmp_max_fm_size=1
        for j in a_size:
            tmp_max_fm_size = tmp_max_fm_size*j 

        if tmp_max_fm_size>max_fm_size:
            max_fm_size = tmp_max_fm_size       
        return max_fm_size
            


