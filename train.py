import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50
from nets.googlenet import googlenet
from nets.vgg import vgg16, vgg16_bn
from nets.vit import vit_b_16
from nets.swin_transformer import swin_transformer_small, swin_transformer_tiny
from nets.swin_transformer_v2 import swin_transformer_V2_tiny
# from utils.eval import eval

get_model_from_name = {
    "mobilenetv2"               : mobilenetv2,
    "resnet50"                  : resnet50,
    "googlenet"                 : googlenet,
    "vgg16"                     : vgg16,
    "vgg16_bn"                  : vgg16_bn,
    "vit_b_16"                  : vit_b_16,
    "swin_transformer_small"    : swin_transformer_small,
    "swin_transformer_tiny"     : swin_transformer_tiny,
    "swin_transformer_V2_tiny"       : swin_transformer_V2_tiny
}

from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator
from utils.utils import get_classes, show_config, get_lr_scheduler, set_optimizer_lr
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    #------------------------------------------------#
    classes_path    = 'model_data/cls_names.txt'
    #------------------------------------------------#
    #   input image shape
    #------------------------------------------------#
    input_shape     = [128,128]
    #------------------------------------------------#
    #   models：
    #   vgg16、vgg16_bn
    #   mobilenetv2
    #   googlenet
    #   resnet50
    #   vit_b_16
    #   swin_transformer_small swin_transformer_tiny
    #   swin_transformer_V2_tiny
    #------------------------------------------------#
    backbone        = "swin_transformer_tiny"
    #------------------------------------------------#
    #   pretrained weight file path
    #------------------------------------------------#
    model_path      = "model_data/swin_tiny.pth"
    #------------------------------------------------#
    #------------------------------------------------#
    device = torch.device('cuda:0')
    #------------------------------------------------#
    #   foldrs of weight parameters and log files
    #------------------------------------------------#
    save_dir        = 'logs'   #checkpoint save path
    model_save_dir  = os.path.join(save_dir,backbone)
    #------------------------------------------------#
    #   train_annotation_path
    #------------------------------------------------#
    train_annotation_path       = './dataset_path/train.txt'
    val_annotation_path         = './dataset_path/val.txt'
    #----------------------------------------------------------------#
    #   training hyper-parameters：Epoch、batchsize、learning rate
    #----------------------------------------------------------------#
    Epoch           = 50
    batch_size      = 256
    init_lr         = 0.01
    min_lr          = init_lr * 0.001
    num_workers     = 16
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    with open(os.path.join(model_save_dir,'configuration.txt'), 'a') as f:
        f.write('backbone = %s\n' %backbone)
        f.write('model_path = %s\n' %model_path)
        f.write('input_shape = %s\n' %input_shape)
        f.write('Epoch = %s\n' %Epoch)
        f.write('batch_size = %s\n' %batch_size)
        f.write('init_lr = %s\n' %init_lr)
    #------------------------------------------------#

    #------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    #------------------------------------------------#
    #   model initialization
    #------------------------------------------------#
    if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small',
                        'swin_transformer_base', 'swin_transformer_V2_tiny']:
        model       = get_model_from_name[backbone](num_classes=num_classes)

    else:
        model       = get_model_from_name[backbone](input_shape=input_shape,num_classes=num_classes)
    model_dict  = model.state_dict()
    print('\033[1;33;44mLoad weights from : {}\033[0m'.format(model_path))
    pretrained_dict = torch.load(model_path)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict,strict=False)
    #------------------------------------------------#
    #   display the no matched key
    #------------------------------------------------#
    print("\nSuccessful Load Key:", str(load_key)[:50], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:50], "……\nFail To Load Key num:", len(no_load_key))
    print("\n\033[1;33;44m温馨提示,head部分没有载入是正常现象,Backbone部分没有载入是错误的。\033[0m")
    # ------------------------------------------------#
    #    record Loss
    # ------------------------------------------------#
    loss_history = LossHistory(backbone,model_save_dir)
    #------------------------------------------------#
    #   read the corresponding txt
    #------------------------------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    # val_lines = train_lines[:int(len(train_lines)*0.2)] # 20%
    # train_lines = train_lines[int(len(train_lines)*0.2):] # 80%
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    #------------------------------------------------#
    #   display hyper-parameters
    #------------------------------------------------#
    show_config(
        num_classes =num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
        Epoch = Epoch, batch_size = batch_size, init_lr = init_lr,\
        save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )
    
    #------------------------------------------------#
    #   adaptively learning rate according to the batchsize
    #------------------------------------------------#
    nbs             = 64
    if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small',
                    'swin_transformer_base','swin_transformer_V2_tiny']:
        nbs         = 256
    lr_limit_max    = 1e-2 #default 1e-2
    lr_limit_min    = 1e-4 #default 1e-4
    Init_lr_fit     = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    
    optimizer       = optim.SGD(model.parameters(),init_lr, momentum=0.9)
    cudnn.benchmark = True
    #------------------------------------------------#
    #   obtain the formula of learning rate decline------cosine annealing,warm up--#
    lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Min_lr_fit, Epoch)
    #------------------------------------------------#
    #   dataload
    #------------------------------------------------#
    train_dataset   = DataGenerator(train_lines,input_shape=input_shape,train=True)
    val_dataset     = DataGenerator(val_lines,input_shape=input_shape,train=False)
    gen             = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True)
    gen_val         = DataLoader(val_dataset,shuffle=True ,batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True)
    
    for epoch in range(Epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(device, model, loss_history, optimizer, epoch, gen, gen_val, Epoch, model_save_dir,backbone)
