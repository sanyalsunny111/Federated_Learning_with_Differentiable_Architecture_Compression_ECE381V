'''Train CIFAR10 with PyTorch.'''
#copied from https://github.com/kuangliu/pytorch-cifar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

import torchvision
import torchvision.transforms as transforms

import shutil, os, copy, json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from thop import profile
from my_model import mbn2_search
from my_model import mbn2_nas

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--path_dir', default='test_dir', type=str, help='log dir')
parser.add_argument('--wm', default=1.0, type=float, help='width_multiplier')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--penalty_factor', default=0.1, type=float, help='entropy loss coefficient')


args = parser.parse_args()

def prepare_data(args):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def create_model(args):
    path_root='./'
    path_dir=args.path_dir+'_mbn2_wm_'+str(args.wm)


    path=path_root+'checkpoint/'+path_dir

    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        print('this dir already exists, please create and use a new dir')
        print('exiting......')
        exit(0)
    logs_file=open(path+'/train.logs','a+')
    shutil.copy(path_root+'mbn2_main.py', path)
    shutil.copy(path_root+'my_model/mbn2_channel.py', path)
    shutil.copy(path_root+'my_model/mbn2_channel_wise.py', path)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch



    # Model
    print('==> Building model..')

    net = mbn2_search.mbn2_nas()
    initial_net = copy.deepcopy(net)
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(net, path+'/initial.pth')
    # net = mobilenetv2.MobileNetV2()

    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = Trues

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        ckpt_path='/home/guihong/fed_nas/nas/checkpoint/try_13_mbn2_wm_0.5/'
        assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(ckpt_path+'ckpt_210.pth')# v17:ckpy-36   other version ckpt-289
        print(checkpoint['net'].keys())
        for key_name in checkpoint['net'].keys():
            print(key_name)
            if '_conv_stem' in key_name:
                pass
            else:
                net.state_dict()[key_name].copy_(checkpoint['net'][key_name])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']+1
        print(best_acc)    

    net = net.to(device)

    return net, path, start_epoch, best_acc, device
trainloader, testloader, classes=prepare_data(args)
net, path, start_epoch, best_acc, device=create_model(args)
def train(epoch, optimizer, scheduler):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_num = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # kk=F.softmax(net.layers[0].conv1_wgt.weight)
        # weight_l1_norm = net.layers[0].conv1_wgt.weight.norm(p=1)
        # print(weight_l1_norm)
        loss = criterion(outputs, targets)+net.get_entropy_loss()*args.penalty_factor
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        batch_num = batch_idx
        if train_loss <3000000: 
            pass
        else: 
            print('infinitus found'+'\n\n')
            exit(0)

    
    # print(net.layers[0].get_entropy_loss())
    with open(path+'/train.logs','a+') as logs_file:
        print('Epoch: %d | Train Loss: %.6f | Reg_loss: %.6f | Acc: %.3f%%  | (%d/%d)'
            %(epoch, train_loss/(batch_num+1), net.get_entropy_loss(), 100.*correct/total, correct, total),end=' || ', file = logs_file)
    print('Epoch: %d | Train Loss: %.6f | Reg_loss: %.6f | Acc: %.3f%%  | (%d/%d)'
          %(epoch, train_loss/(batch_num+1), net.get_entropy_loss(), 100.*correct/total, correct, total))

    #'''
    '''
    lr = 0.0001
    penalty_factor =1
    if epoch < 50:
        lr=0.1
    elif epoch < 100:
        lr=0.01
    elif epoch < 150:
        lr=0.001

    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0, weight_decay=5e-4)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # kk=F.softmax(net.layers[0].conv1_wgt.weight)
        # weight_l1_norm = net.layers[0].conv1_wgt.weight.norm(p=1)
        # print(weight_l1_norm)
        loss = criterion(outputs, targets)+net.get_entropy_loss()*penalty_factor
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        batch_num = batch_idx
        if train_loss <3000000: 
            pass
        else: 
            print('\n\n')
            exit(0)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # print(net.layers[0].get_entropy_loss())
    print('Epoch: %d | Train Loss: %.6f | Reg_loss: %.6f | Acc: %.3f%%  | (%d/%d)'
          %(epoch, train_loss/(batch_num+1), net.get_entropy_loss(), 100.*correct/total, correct, total),end=' || ', file = logs_file)
    print('Epoch: %d | Train Loss: %.6f | Reg_loss: %.6f | Acc: %.3f%%  | (%d/%d)'
          %(epoch, train_loss/(batch_num+1), net.get_entropy_loss(), 100.*correct/total, correct, total))
    #'''

    # if isinstance(net,mbn2_channel_wise) and epoch==39:
    #     batch_mean(net)     
    # if batch_idx == 0 :
    #     break

    # exit(0)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_num = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #print(inputs.size())
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_num = batch_idx
    with open(path+'/train.logs','a+') as logs_file:
        print('Test  Loss: %.6f | Acc: %.3f%%  | (%d/%d)'
            % (test_loss/(batch_num+1), 100.*correct/total, correct, total),end='\n', file = logs_file)

    print('Test  Loss: %.6f | Acc: %.3f%%  | (%d/%d)'
          % (test_loss/(batch_num+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, path+'/ckpt_'+str(epoch)+'.pth')
        best_acc = acc
    elif epoch%5==0:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, path+'/ckpt_'+str(epoch)+'.pth')
    # if epoch%15==0:
    #     net._apply_mask()

def resume_wgt(net):
    net.generate_mask()
    # conv_tmp_list=[]
    # for block in net.layers:
    #     conv_tmp_list.append([block.conv1_one, block.conv2_one, block.conv1_mask, block.conv2_mask])

    # net = torch.load(path+'/initial.pth')
    # net.eval()
    # for i, block in enumerate(net.layers):
    #     block.conv1_one, block.conv2_one, block.conv1_mask, block.conv2_mask=conv_tmp_list[i][0],\
    #             conv_tmp_list[i][1], conv_tmp_list[i][2], conv_tmp_list[i][3]
    net = net.to(device)
    # print(net.conv1.weight)
    return net


def zero_count(wgt_mat):
    wgt=wgt_mat.view(-1)
    # print(wgt)
    zero_num=(torch.abs(wgt) < 0.1).sum(dim=0)
    return zero_num


cfg_dict = {}

def extract_arch(idx):
    profile_input = torch.randn(1, 3, 32, 32)
    def extract_single_net(net):
        cfg = [[1*32,  16,  1],
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

        in_channels_dict=np.zeros((17,4))
        out_channels_dict=np.zeros((17,4))
        kernel_size_dict=np.zeros((17,4))
        zeros_num=np.zeros((17,4))
        for i in range(17):
            in_channels_dict[i,0]=net.layers[i].conv1.in_channels
            in_channels_dict[i,1]=net.layers[i].conv2.in_channels
            in_channels_dict[i,2]=net.layers[i].conv3.in_channels

            out_channels_dict[i,0]=net.layers[i].conv1.out_channels
            out_channels_dict[i,1]=net.layers[i].conv2.out_channels
            out_channels_dict[i,2]=net.layers[i].conv3.out_channels

            kernel_size_dict[i,0]=(net.layers[i].conv1.kernel_size[0])**2
            kernel_size_dict[i,1]=(net.layers[i].conv2.kernel_size[0])**2
            kernel_size_dict[i,2]=(net.layers[i].conv3.kernel_size[0])**2

            zeros_num[i,0]=(zero_count(net.layers[i].conv1_wgt.weight.data))
            zeros_num[i,1]=(zero_count(net.layers[i].conv2_wgt.weight.data))
            zeros_num[i,2]=(zero_count(net.layers[i].conv3_wgt.weight.data))
            
            if (len(net.layers[i].shortcut))>0:
                in_channels_dict[i,3]=net.layers[i].shortcut[0].in_channels
                out_channels_dict[i,3]=net.layers[i].shortcut[0].out_channels
                kernel_size_dict[i,3]=(net.layers[i].shortcut[0].kernel_size[0])**2
                zeros_num[i,3]=(zero_count(net.layers[i].short_wgt.weight.data))

        for i in range(len(cfg)):
            cfg[i][0] = max(int(cfg[i][0] - zeros_num[i,0]), 8)
        
        return cfg

    net_config = {}
    cfg = extract_single_net(net)

    nas_net = mbn2_nas.mbn2_nas(cfg)

    macs, params = profile(nas_net, inputs=(profile_input, ))
    net_config['macs'] = macs
    net_config['params'] = params
    net_config['cfg'] = cfg
    cfg_dict[int(idx)] = net_config




criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=0.0)
for epoch in range(start_epoch, args.epochs):
    # resume_wgt()
    train(epoch, optimizer, scheduler)
    test(epoch)
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
with open('net_cfg.json', 'w+') as f:
    json.dump(cfg_dict, f) 
exit(0)
