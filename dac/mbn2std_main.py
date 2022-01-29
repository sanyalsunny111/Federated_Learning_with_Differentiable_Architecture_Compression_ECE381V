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

import shutil, os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from my_model import mbn2_std

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--path_dir', default='scratch_loss_v17', type=str, help='log dir')
parser.add_argument('--wm', default=1.0, type=float, help='width_multiplier')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--penalty_factor', default=1.0, type=float, help='weight decay')

args = parser.parse_args()

# from thop import profile
# for i in range(100):
#     model = mbn2_std.mbn2(width_mul=(i+1.0)/100)
#     input = torch.randn(1, 3, 32, 32)
#     macs, params = profile(model, inputs=(input, ))
#     with open('1.logs', 'a+') as file_name:
#         print((i+1.0)/100, macs, params, file=file_name)
# exit(0)

path_root='/work/06765/ghl/project/fed_nas/nas/'
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
    root='/work/06765/ghl/project/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='/work/06765/ghl/project/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()~
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()



net = mbn2_std.mbn2(width_mul=args.wm)
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
best_acc = 0  # best test accuracy


# Training
def net_prune(net,threhold=0.001):

    net.to(device)
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
        loss = criterion(outputs, targets)
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

    
    print('Epoch: %d | Train Loss: %.6f | Acc: %.3f%%  | (%d/%d)'
          %(epoch, train_loss/(batch_num+1), 100.*correct/total, correct, total),end=' || ', file = logs_file)
    print('Epoch: %d | Train Loss: %.6f | Acc: %.3f%%  | (%d/%d)'
          %(epoch, train_loss/(batch_num+1), 100.*correct/total, correct, total))

    
    


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


    print('Test  Loss: %.6f | Acc: %.3f%%  | (%d/%d)'
          % (test_loss/(batch_num+1), 100.*correct/total, correct, total),end='\n', file = logs_file)


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



optimizer = torch.optim.SGD(net.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=0.0)
for epoch in range(start_epoch, start_epoch+args.epochs):

    train(epoch, optimizer, scheduler)
    test(epoch)
    scheduler.step()
    lr = scheduler.get_last_lr()[0]


exit(0)
