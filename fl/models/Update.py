#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, gpu_id=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if gpu_id:
            self.args.device = 'cuda:{}'.format(gpu_id)
    def cal_entropy(self,array):
        unique_data = np.unique(array)
        resdata = []
        for ii in unique_data:
            resdata.append(sum(array == ii))
        total_num=np.sum(np.array(resdata))
        freq=np.array(resdata,dtype=float)
        freq=freq/total_num
        entropy=np.sum(-np.log(freq)*freq)
        return entropy,total_num,len(resdata)

    def train(self, net,random_epoch=False):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        local_epochs=self.args.local_ep
        # print(random_epoch)
        if random_epoch==True:
            local_epochs=random.randint(int(self.args.local_ep/3),int(self.args.local_ep))

        global_correct=0
        global_total = 0
        global_local_entropy,global_sample_num,global_num_classes, global_num_params = 0, 0, 0, 0
        for iter in range(local_epochs):
            batch_loss = []
            all_labels=[]
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                _, predicted = log_probs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                if batch_idx==0:
                    all_labels=labels.detach().cpu().numpy()
                else:
                    all_labels=np.concatenate((all_labels,labels.detach().cpu().numpy()))
            local_entropy,sample_num,num_classes=self.cal_entropy(all_labels)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            num_params=sum(p.numel() for p in net.parameters())
            global_num_params= num_params
            global_correct=correct
            global_total = total
            global_local_entropy,global_sample_num,global_num_classes = local_entropy,sample_num,num_classes
        return net.state_dict(), sum(epoch_loss), len(epoch_loss),100.*global_correct, global_total, global_local_entropy,global_sample_num,global_num_classes,local_epochs,global_num_params

