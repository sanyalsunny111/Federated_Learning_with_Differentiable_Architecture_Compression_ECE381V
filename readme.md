
# DAC 
## Search
- search the optimal network architecture given differrent MACs and #Params budget
- usage: 
    - python dac_search.py  [arguments]

| optional arguments | Description |
| ----------- | ----------- |
|  --lr     |           learning rate |
|  --resume, -r   |       resume from checkpoint |
|  --epochs    |    number of training epochs |
|  --batch_size | batch size |
|  --path_dir  |  log dir |
|  --wm         |       width_multiplier |
|  --momentum   | momentum |
|  --weight_decay |  weight decay |
|  --penalty_factor | entropy loss coefficient |


    * python train_mlp.py --depth=8 --width=8 --tc=10 --dataset='MNIST' 


## Train the searched network
    - python dac_train.py  [arguments]


| optional arguments | Description |
| ----------- | ----------- |
|   --lr           |       learning rate | 
|   --resume, -r      |      resume from checkpoint | 
|   --epochs      |    number of training epochs | 
|   --batch_size  |  batch size | 
|   --path_dir   |   log dir | 
|   --wm         |         width_multiplier | 
|   --momentum   |   momentum | 
|   --weight_decay  |    weight decay | 
|   --json_name  |   network configs file | 
|   --net_config_id  |   network configs index | 



# Federated Learning
- usage: 
    - python main_fed.py  [arguments]

| optional arguments | Description |
| ----------- | ----------- |
|  --epochs  | rounds of training |
|  --num_users |  number of users: K |
|  --local_models |  number of local models used for merging |
|  --frac   | the fraction of clients: C |
|  --local_ep  | the number of local epochs: E |
|  --local_bs  | local batch size: B |
|  --bs   | test batch size |
|  --lr  |  learning rate |
|  --momentum  | SGD momentum (default: 0.5) |
|  --split   | train-test split type, user or sample |
|  --model   | model name |
|  --cfg   | network configurations (only for mbn2) |
|  --macs_bugget  |  macs bugget/constraints (only for mbn2) |
|  --params_bugget  | #params bugget/constraints (only for mbn2) |
|  --json_name   | list of network configuration definition (only for mbn2) |
|  --kernel_num    |  number of each kind of kernel |
|  --kernel_sizes    |  comma-separated kernel size to use for convolution |
|  --norm  |  batch_norm, layer_norm, or None |
|  --num_filters    |  number of filters for conv nets |
|  --max_pool |  Whether use max pooling rather than strided convolutions |
|  --dataset |  name of dataset |
|  --iid   |  whether i.i.d or not |
|  --num_classes |  number of classes |
|  --num_channels  | number of channels of images |
|  --gpu  |  GPU ID, -1 for CPU |
|  --stopping_rounds  | rounds of early stopping |
|  --verbose  |  verbose print |
|  --seed  |  random seed (default: 1) |
|  --random_epoch  | random epochs of local training or not |
|  --save_ckpt  |  save checkpoint |
|  --log_dir  |  path of saving checkpoint |
