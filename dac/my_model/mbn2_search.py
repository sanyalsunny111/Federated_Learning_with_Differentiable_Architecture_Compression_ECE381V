import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


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

        self.conv1_wgt=torch.nn.Conv2d(in_channels=inter_planes, out_channels=inter_planes, kernel_size=1, stride=1,  groups=inter_planes, bias=False)
        self.conv2_wgt=torch.nn.Conv2d(in_channels=inter_planes, out_channels=inter_planes, kernel_size=1, stride=1,  groups=inter_planes, bias=False)
        self.conv3_wgt=torch.nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=1, stride=1,  groups=out_planes, bias=False)
        self.short_wgt=torch.nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=1, stride=1,  groups=out_planes, bias=False)
        self.conv1_wgt.weight.data=torch.ones(inter_planes,1,1,1)
        self.conv2_wgt.weight.data=torch.ones(inter_planes,1,1,1)
        self.conv3_wgt.weight.data=torch.ones(out_planes,  1,1,1)
        self.short_wgt.weight.data=torch.ones(out_planes,  1,1,1)

        self.conv1_mask=None 
        self.conv2_mask=None 
        self.conv1_one=range(inter_planes)
        self.conv2_one=range(inter_planes)

    def forward(self, x):
        a = F.relu(self.bn1(self.conv1(x)))
        a = self.conv1_wgt(a)
        if self.conv1_mask:
            a[:,self.conv1_mask,:,:]=0

        a = F.relu(self.bn2(self.conv2(a)))
        if self.conv1_mask:
            a[:,self.conv1_mask,:,:]=0

        a = self.bn3(self.conv3(a))
        if self.stride==1 :
            b = self.shortcut(x)
            a=a+b
            return a
        else:
            return a 
    def generate_mask(self, threshold=0.01):
        if self.conv1_mask==None:
            self.conv1_mask=torch.squeeze(self.conv1_wgt.weight.data.cpu()).numpy()
            # print(self.conv1_mask)
            # print((np.abs(self.conv1_mask) >= threshold).nonzero())
            self.conv1_one = (np.abs(self.conv1_mask) >= threshold).nonzero()[0].tolist()
            self.conv1_mask=(np.abs(self.conv1_mask) < threshold).nonzero()[0].tolist()

            self.conv2_mask=torch.squeeze(self.conv2_wgt.weight.data.cpu()).numpy()
            self.conv2_one = (np.abs(self.conv2_mask) >= threshold).nonzero()[0].tolist()
            self.conv2_mask=(np.abs(self.conv2_mask) < threshold).nonzero()[0].tolist()
        else:
            conv1_mask=torch.squeeze(self.conv1_wgt.weight.data.cpu()).numpy()
            conv1_one = (np.abs(self.conv1_mask) >= threshold).nonzero()[0].tolist()
            conv1_mask=(np.abs(self.conv1_mask) < threshold).nonzero()[0].tolist()

            conv2_mask=torch.squeeze(self.conv2_wgt.weight.data.cpu()).numpy()
            conv2_one = (np.abs(self.conv2_mask) >= threshold).nonzero()[0].tolist()
            conv2_mask=(np.abs(self.conv2_mask) < threshold).nonzero()[0].tolist()
            
            for i in conv1_mask:
                if i in self.conv1_mask:
                    pass
                else:
                    self.conv1_mask.append(i)
            for i in conv2_mask:
                if i in self.conv2_mask:
                    pass
                else:
                    self.conv2_mask.append(i)

            temp_conv1_one=[]
            temp_conv2_one=[]

            for i in conv1_one:
                if i in self.conv1_one:
                    temp_conv1_one.append(i)
            for i in conv2_one:
                if i in self.conv2_one:
                    temp_conv1_one.append(i)
            self.conv1_one = temp_conv1_one
            self.conv2_one = temp_conv2_one
        self.conv1_wgt.weight.data[self.conv1_mask]=0
    def get_entropy_loss(self):
        entropy_loss = 0
        entropy_loss = entropy_loss - torch.sum(torch.squeeze(torch.mul(F.softmax(self.conv1_wgt.weight[self.conv1_one],dim=0),\
                                         torch.log(F.softmax(self.conv1_wgt.weight[self.conv1_one],dim=0)))))


        margin_loss=0
        conv1_zeros=torch.zeros_like(self.conv1_wgt.weight.data)
        conv1_ones =torch.ones_like(self.conv1_wgt.weight.data)
        conv1_mask=torch.where(self.conv1_wgt.weight.data<0.0,conv1_ones,conv1_zeros)
        margin_loss=margin_loss+torch.sum(torch.mul(conv1_zeros-self.conv1_wgt.weight,conv1_mask))
        conv1_mask=torch.where(self.conv1_wgt.weight.data>1.0,conv1_ones,conv1_zeros)
        margin_loss=margin_loss+torch.sum(torch.mul(self.conv1_wgt.weight-conv1_ones,conv1_mask))


        entropy_loss=entropy_loss+0.1*margin_loss
        return entropy_loss

class mbn2_nas(nn.Module):
    
    def __init__(self,cfg=None, num_classes=10):
        super(mbn2_nas, self).__init__()
        if cfg:
            self.cfg=cfg
        else:
            # (inter_planes, out_planes, stride)
            self.cfg = \
            [[1*32,  16,  1],
            [6*16,  24,  1], # NOTE: change stride 2 -> 1 for CIFAR10
            [6*24,  24,  1],
            [6*24,  32,  2],
            [6*32,  32,  1],
            [6*32,  32,  1],
            [6*32,  64,  2],
            [6*64,  64,  1],
            [6*64,  64,  1],
            [6*64,  64,  1],
            [6*64,  96,  1],
            [6*96,  96,  1],
            [6*96,  96,  1],
            [6*96, 160, 2],
            [6*160, 160, 1],
            [6*160, 160, 1],
            [6*160, 320, 1]]
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
    def get_entropy_loss(self):
        entropy_loss = 0
        for block_layer in self.layers:
            entropy_loss = entropy_loss + block_layer.get_entropy_loss()
        return entropy_loss

    def generate_mask(self):
        with torch.no_grad():
            
            for block_layer in self.layers:
                block_layer.generate_mask()

