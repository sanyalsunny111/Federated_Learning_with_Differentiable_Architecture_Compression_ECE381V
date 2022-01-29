import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

global batch_size
batch_size = 100
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.in_planes=in_planes
        self.planes=planes
        self.out_planes=out_planes
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        self.expansion = expansion


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



class mbn2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, width_mul=1.0, round_nearest=8, num_classes=10,batch_sizes=128):
        super(mbn2, self).__init__()
        global batch_size
        batch_size=batch_sizes
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10

        input_channel=_make_divisible(32 * width_mul, round_nearest)
        out_channel=_make_divisible(1280 * max(1.0, width_mul), round_nearest)
        last_input_channel=_make_divisible(320 * width_mul, round_nearest)

        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        # self.conv1_channel_wgt = nn.Parameter(torch.ones(batch_size,32),requires_grad=True)

        self.layers = self._make_layers(in_planes=input_channel,width_mul=width_mul, round_nearest=round_nearest)
        self.conv2 = nn.Conv2d(last_input_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.linear = nn.Linear(out_channel, num_classes)


    def _make_layers(self, in_planes, width_mul, round_nearest):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            out_planes=_make_divisible(out_planes * width_mul, round_nearest)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
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
            tmp_max_fm_size=block.expansion
            for j in a_size:
                tmp_max_fm_size = tmp_max_fm_size*j 
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
            

