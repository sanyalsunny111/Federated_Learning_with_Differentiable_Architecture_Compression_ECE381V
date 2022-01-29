#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np



def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
#'''

'''
def FedAvg(w,model_num,metrics):
    w_avg = copy.deepcopy(w[0])
    high_metrics=np.argsort(metrics)
    high_metrics=high_metrics[::-1]
    print(metrics)
    print(high_metrics)
    for k in w_avg.keys():
        for i in range(model_num):
            if i == 0:
                w_avg[k] = w[high_metrics[i]][k] 
            else:
                w_avg[k] += w[high_metrics[i]][k]
        w_avg[k] = torch.div(w_avg[k], model_num)
    return w_avg
#'''

