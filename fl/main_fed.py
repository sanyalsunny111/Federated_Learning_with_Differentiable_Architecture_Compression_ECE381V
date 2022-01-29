import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import json
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_10_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar,mbn2_nas
from models.Fed import FedAvg
from models.test import test_img
import datetime,shutil,os
import threading,time


def simultaneous_local_run(epoch, args, dataset_train, dict_users, net_glob, ckpt_path):
    gpu_count = torch.cuda.device_count()
    w_locals, loss_locals, local_epochs,model_data_entropy,model_data_number = [], [], [], [], []
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    feat_array=np.zeros((m,7))
    i=0
    for i in range(m):
        w_locals.append([])
        loss_locals.append([])
        local_epochs.append([])
        model_data_entropy.append([])
        model_data_number.append([])

    def single_local_device(i, device_idx, gpu_id):
        
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[device_idx])
        #net.state_dict(), sum(epoch_loss) / len(epoch_loss),100.*correct/total,local_entropy,sample_num,num_classes,local_epochs,num_params
        w, local_loss,num_loss, local_acc, local_total, local_entropy,sample_num,num_classes, local_epoch,num_params = local.train(net=copy.deepcopy(net_glob).to(args.device),random_epoch=args.random_epoch)
        loss_locals[i]=copy.deepcopy(local_loss)
        local_epochs[i]=copy.deepcopy(local_epoch)
        feat_array[i]=np.array([local_loss,local_acc,local_entropy,sample_num,num_classes, local_epoch,num_params])

        if args.save_ckpt:
            print('Saving..')
            state = {
                'net': w,
                'epoch': iter,
            }
            torch.save(state, ckpt_path+'/user_idx_'+str(device_idx)+'.pth')
        w_locals[i]=copy.deepcopy(w)
        model_data_entropy[i]=local_entropy
        model_data_number[i]=sample_num
        #'''
        time.sleep(device_idx)
        print(i, device_idx)
    threads_list = []
    print(idxs_users)
    
    for i,device_idx in enumerate(idxs_users):
        single_local_device(i, device_idx, 0)
    '''
        try:
            print('normal', i, device_idx)
            t = threading.Thread(target=single_local_device, args=(i, device_idx, int(i%gpu_count)))
            threads_list.append(t)
            t.start()
        except:
            print('possible error: CUDA out of memory', i, device_idx)
            continue

    for t in threads_list:
        t.join()
    #'''
    loss_locals=np.array(loss_locals)
    local_epochs=np.array(local_epochs)
    model_data_entropy=np.array(model_data_entropy)
    model_data_number=np.array(model_data_number)

    mean_data_number=np.mean(model_data_number)
    mean_data_entropy=np.mean(model_data_entropy)

    #model_metric=np.log(model_data_number)*model_data_entropy*(-loss_locals)
    model_metric=np.log(model_data_number)*model_data_entropy
    np.savetxt(ckpt_path+'feat.logs',feat_array)
    w_glob = FedAvg(w_locals)
    net_glob.load_state_dict(w_glob)
    return net_glob, loss_locals, local_epochs

'''
def fed_main(args):
    # parse args
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    path='ckpt/'+args.log_dir

    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path+'/fig/'):
        os.mkdir(path+'/fig/')
    # else:
    #     print('this dir already exists, please create and use a new dir')
    #     print('exiting......')
    #     exit(0)
    shutil.copy('main_fed.py', path)
    shutil.copy('models/Fed.py', path)
    shutil.copy('models/Update.py', path)
    model_data_entropy=[]
    model_data_number=[]
    model_local_epoch=[]
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('/work/06765/ghl/project/data', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('/work/06765/ghl/project/data', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_10_noniid(dataset_train, args.num_users)
            # exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'mbn2':
        net_config_dict=json.load(open(args.json_name,'r'))
        for net_name in net_config_dict.keys():
            if net_config_dict[net_name]['macs']<args.macs_bugget \
                    and net_config_dict[net_name]['params']<args.macs_bugget:
                net_cfg = net_config_dict[net_name]['cfg']
        if net_cfg:
            net_glob = mbn2_nas(net_cfg).to(args.device)
        else:
            print('too strict macs/params constraints')
            exit(0)
    else:
        exit('Error: unrecognized model')
    # print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    loss_file_name=path+'/loss.logs'
    epoch_file_name=path+'/epoch.logs'
    acc_file_name=path+'/acc.logs'
    with open(loss_file_name,'a+') as file_name:
        print(args,file=file_name)
        print(datetime.datetime.now(),file=file_name)
        print('dataset:',args.dataset,'models:',args.model,'epochs:',args.epochs,'local_ep:', args.local_ep,'iid:',args.iid,'random epochs:',args.random_epoch,'number_users:',args.num_users,'frac_users',args.frac,'merged local models:',args.local_models,file=file_name)
    with open(epoch_file_name,'a+') as file_name:
        print(datetime.datetime.now(),file=file_name)
        print('dataset:',args.dataset,'models:',args.model,'epochs:',args.epochs,'local_ep:', args.local_ep,'iid:',args.iid,'random epochs:',args.random_epoch,'number_users:',args.num_users,'frac_users',args.frac,'merged local models:',args.local_models,file=file_name)

    for iter in range(args.epochs):
        ckpt_path=path+'/ckpt/round_'+str(iter)
        if not os.path.isdir(ckpt_path):
            if not os.path.isdir(path+'/ckpt'):
                os.mkdir(path+'/ckpt')
            os.mkdir(ckpt_path)

        net_glob, loss_locals, local_epochs=simultaneous_local_run(iter, args, dataset_train, dict_users, net_glob, ckpt_path)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        with open(loss_file_name,'a+') as file_name:
            print(loss_avg,'local',loss_locals,file=file_name)
        with open(epoch_file_name,'a+') as file_name:
            print(local_epochs,file=file_name)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(path+'/fig/fed_{}_{}_{}_C{}_iid_{}_randomepochs_{}_localmodels{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.random_epoch,args.local_models))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    with open(acc_file_name,'a+') as file_name:
        print('dataset:',args.dataset,'models:',args.model,'epochs:',args.epochs,'iid:',args.iid,'random epochs:',args.random_epoch,'number_users:',args.num_users,'frac_users',args.frac,'merged local models:',args.local_models,end='   ',file=file_name)
        print("Training accuracy: {:.2f}".format(acc_train),end='   ', file=file_name)
        print("Testing accuracy: {:.2f}".format(acc_test),file=file_name)
#'''
def fed_main(args):
    # parse args
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    path_root='./'
    path=path_root+'ckpt/'+args.log_dir
    # os.system('cp -r /work/06765/ghl/project/data path')
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path+'/fig/'):
        os.mkdir(path+'/fig/')
    # else:
    #     print('this dir already exists, please create and use a new dir')
    #     print('exiting......')
    #     exit(0)
    shutil.copy(path_root+'main_fed.py', path)
    shutil.copy(path_root+'models/Fed.py', path)
    shutil.copy(path_root+'models/Update.py', path)
    model_data_entropy=[]
    model_data_number=[]
    model_local_epoch=[]
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(path+'/data', train=True, download=True, transform=trans_cifar)
        dataset_test  = datasets.CIFAR10(path+'/data', train=False, download=True, transform=trans_cifar)
        # print(dataset_train.targets)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_10_noniid(dataset_train, args.num_users)
            # exit('Error: only consider IID setting in CIFAR10')

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'mbn2':
        net_config_dict=json.load(open(args.json_name,'r'))
        for net_name in net_config_dict.keys():
            if net_config_dict[net_name]['macs']<args.macs_bugget \
                    and net_config_dict[net_name]['params']<args.params_bugget:
                net_cfg = net_config_dict[net_name]['cfg']
                break
        print(net_config_dict[net_name]['macs'], net_config_dict[net_name]['params'])
        if net_cfg:
            net_glob = mbn2_nas(net_cfg).to(args.device)
        else:
            print('too strict macs/params constraints')
            exit(0)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    loss_file_name=path+'/loss.logs'
    epoch_file_name=path+'/epoch.logs'
    acc_file_name=path+'/acc.logs'
    with open(loss_file_name,'a+') as file_name:
        print(args,file=file_name)
        print(datetime.datetime.now(),file=file_name)
        print('dataset:',args.dataset,'models:',args.model,'epochs:',args.epochs,'iid:',args.iid,'random epochs:',args.random_epoch,'number_users:',args.num_users,'frac_users',args.frac,'merged local models:',args.local_models,file=file_name)
    with open(epoch_file_name,'a+') as file_name:
        print(datetime.datetime.now(),file=file_name)
        print('dataset:',args.dataset,'models:',args.model,'epochs:',args.epochs,'iid:',args.iid,'random epochs:',args.random_epoch,'number_users:',args.num_users,'frac_users',args.frac,'merged local models:',args.local_models,file=file_name)

    for iter in range(args.epochs):
        ckpt_path=path+'/ckpt/round_'+str(iter)
        if not os.path.isdir(ckpt_path):
            if not os.path.isdir(path+'/ckpt'):
                os.mkdir(path+'/ckpt')
            os.mkdir(ckpt_path)

        w_locals, loss_locals, local_epochs,model_data_entropy,model_data_number = [], [], [], [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        feat_array=np.zeros((m,7))
        i=0
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            #net.state_dict(), sum(epoch_loss) / len(epoch_loss),100.*correct/total,local_entropy,sample_num,num_classes,local_epochs,num_params
            w, local_loss, num_local_epoch, local_acc, local_total, local_entropy,sample_num,num_classes, local_epoch,num_params = local.train(net=copy.deepcopy(net_glob).to(args.device),random_epoch=args.random_epoch)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(local_loss))
            local_epochs.append(copy.deepcopy(local_epoch))
            feat_array[i]=np.array([local_loss,local_acc,local_entropy,sample_num,num_classes, local_epoch,num_params])

            if args.save_ckpt:
                print('Saving..')
                state = {
                    'net': w,
                    'epoch': iter,
                }
                torch.save(state, ckpt_path+'/user_idx_'+str(i)+'.pth')

            model_data_entropy.append(local_entropy)
            model_data_number.append(sample_num)
            i=i+1
        np.savetxt(ckpt_path+'feat.logs',feat_array)
        # update global weights
        loss_locals=np.array(loss_locals)
        local_epochs=np.array(local_epochs)
        model_data_entropy=np.array(model_data_entropy)
        model_data_number=np.array(model_data_number)

        mean_data_number=np.mean(model_data_number)
        mean_data_entropy=np.mean(model_data_entropy)

        #model_metric=np.log(model_data_number)*model_data_entropy*(-loss_locals)
        model_metric=np.log(model_data_number)*model_data_entropy
        # w_glob = FedAvg(w_locals,args.local_models,model_metric)
        w_glob = FedAvg(w_locals)#

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        with open(loss_file_name,'a+') as file_name:
            print(loss_avg,'local',loss_locals,file=file_name)
        with open(epoch_file_name,'a+') as file_name:
            print(local_epochs,file=file_name)

        print('iid:',args.iid,'random epochs:',args.random_epoch,'number_users:',args.num_users)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(path+'/fig/fed_{}_{}_{}_C{}_iid_{}_randomepochs_{}_localmodels{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.random_epoch,args.local_models))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    with open(acc_file_name,'a+') as file_name:
        print(args,  file=file_name)
        print('dataset:',args.dataset,'models:',args.model,'epochs:',args.epochs,'iid:',args.iid,'random epochs:',args.random_epoch,'number_users:',args.num_users,'frac_users',args.frac,'merged local models:',args.local_models,end='   ',file=file_name)
        print("Training accuracy: {:.2f}".format(acc_train),end='   ', file=file_name)
        print("Testing accuracy: {:.2f}".format(acc_test),file=file_name)
def fed_main_scan(args):
    # parse args
    frac = args.frac
    num_steps=20
    for i in range(num_steps-1):
        args.frac = frac*(i+1)/num_steps
        fed_main(args)

args = args_parser()
args.log_dir='e_{}_mac_{}_param_{}_ns_{}_frac_{}_iid_{}_random_{}'.format(args.epochs, int(args.macs_bugget), int(args.params_bugget) ,args.num_users, args.frac, args.iid, args.random_epoch)
fed_main(args)
