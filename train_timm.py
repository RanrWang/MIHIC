'''2023.7
Author: Ranran Wang.'''


import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator
from utils.utils import get_classes, show_config, get_lr_scheduler, set_optimizer_lr
from utils.utils_fit import fit_one_epoch

from lion_pytorch import Lion

''' the usage of timm package:
import timm
# print(timm.list_models()) 
# print(len(timm.list_models()))
# print(timm.list_models(pretrained=True)) # print the name of models which are pretrained.
# print(len(timm.list_models(pretrained=True)))
# print(timm.list_models('*swin_tiny*',pretrained=True))
# print(len(timm.list_models('*swin_tiny*',pretrained=True)))
# model = timm.create_model('swin_tiny_patch4_window7_224',pretrained=True,num_classes=7) # the other usage to load model.
# print(model)
# print(model.default_cfg)
# print(model.get_classifier())
'''



if __name__ == "__main__":
    #------------------------------------------------#
    #------------------------------------------------#
    #   load the model from timm
    from timm.models.swin_transformer import swin_tiny_patch4_window7_224
    from timm.models.coat import coat_tiny
    from timm.models.convnext import convnext_tiny
    from timm.models.convit import convit_tiny
    from timm.models.crossvit import crossvit_tiny_240
    from timm.models.efficientnet import efficientnet_b0
    from timm.models.hrnet import hrnet_w18
    from timm.models.cait import cait_s24_224
    from timm.models.twins import twins_pcpvt_base, twins_svt_base

    # -------------------------------------------------#
    '''
    # coat_tiny: coat_tiny. batchsize=512. img_size can be set as (128,128) or (224,224).
    # swin_timm_tiny: swin_tiny_patch4_window7_224. batchsize=256. For matching the window_size=7, img_size is set as (224,224). 
    The other version perform '_make_divisible function', which is used to ensure that a given value.
    is divisible by a specified divisor, so img_size in the other version is (128,128),window_size=7.
    # convnext_tiny: convnext_tiny. batchsize=512. img_size=128. But it can also be resize to (224,224).
    # convit_tiny: convit_tiny. batchsize=512. sgd. img_size is resize to (224,224). The other parameter is default.
    # efficientnet_b0: efficientnet_b0. batchsize=512.  img_size is (128,128). But it can also be resize to (224,224).
    # hrnet_w18: hrnet_w18. batchsize=256. img_size=128.  But it can also be resize to (224,224).
    # cait: cait_s24_224. batchsize=128. img_size=224. The training time of this model is too long, so it has not been trained.
    # crossvit_tiny: crossvit_tiny_240. batchsize=512. img_size=128. 
    # crossvit_tiny_224: crossvit_tiny_240. batchsize=512. img_size=224.
    # twins_pcpvt_base: twins_papvt_base. batchsize=512. img_size can be set as (128,128)(batchsize=512) or (224,224)(batchsize=256).
    # twins_svt_base: twins_svt_base. img_size can be set as (128,128)(batchsize=512) or (224,224)(batchsize=256).
    '''
    # backbone = 'coat_tiny'
    # backbone = 'swin_timm_tiny'
    backbone = 'convnext_tiny'
    # backbone = 'convit_tiny'
    # backbone = 'efficientnet_b0'
    # backbone = 'hrnet_w18'
    # backbone = 'cait'
    # backbone = 'crossvit_tiny' #crossvit_tiny (128,128)
    # backbone = 'twins_pcpvt_base'
    # backbone = 'twins_svt_base'
    if backbone == 'coat_tiny':
        model = coat_tiny(img_size=128, pretrained=True, num_classes=7)  # set img_size=128
        # if img_size=224, use the following codes:
        # model = coat_tiny(pretrained=True, num_classes=7)
        # input_shape = [224,224]
    if backbone == 'swin_timm_tiny':
        model = swin_tiny_patch4_window7_224(pretrained=True, num_classes=7)
        input_shape = [224, 224]
    if backbone == 'convnext_tiny':
        model = convnext_tiny(pretrained=True, num_classes=7)
    if backbone == 'convit_tiny':
        model = convit_tiny(pretrained=True, num_classes=7)
        input_shape = [224, 224]
    if backbone == 'efficientnet_b0':
        model = efficientnet_b0(pretrained=True, num_classes=7)
    if backbone == 'hrnet_w18':
        model = hrnet_w18(pretrained=True, num_classes=7)
        # input_shape = [224,224]
    if backbone == 'cait':
        model = cait_s24_224(pretrained=True, num_classes=7)
        input_shape = [224, 224]
    if backbone == 'crossvit_tiny':
        model = crossvit_tiny_240(pretrained=True, num_classes=7)
        # input_shape = [224,224] #if crossvit_tiny_224, otherwise comment out.
    if backbone == 'twins_pcpvt_base':
        model = twins_pcpvt_base(pretrained=True, num_classes=7)
        # input_shape = [224,224]
    if backbone == 'twins_svt_base':
        model = twins_svt_base(pretrained=True, num_classes=7)
        # input_shape = [224,224]
    # ----------------------------------------------------------------#
    #   usage of the GPU
    #------------------------------------------------#
    device = torch.device('cuda:0')
    # ------------------------------------------------#
    save_dir = './logs'
    model_save_dir = os.path.join(save_dir,backbone)
    # ------------------------------------------------#
    #   train_annotation_path training images paths and labels
    # ------------------------------------------------#
    train_annotation_path = './dataset_path/train.txt'
    val_annotation_path = './dataset_path/val.txt'

    #   class name txt
    # ------------------------------------------------#
    classes_path = 'model_data/cls_names.txt'
    #------------------------------------------------#
    #   obtain the classes
    # ------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    #------------------------------------------------#
    # ------------------------------------------------#
    input_shape = [128, 128] # Note that it may be changed in the subsequent processing.
    # ------------------------------------------------#

    #----------------------------------------------------------------#
    #   training parameters：Epoch、batchsize、learning rate
    #----------------------------------------------------------------#
    Epoch           = 50
    batch_size      = 256   ###default=256
    init_lr         = 0.01  ###default=0.01
    min_lr          = init_lr * 0.001
    num_workers     = 32
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    with open(os.path.join(model_save_dir,'configuration.txt'), 'a') as f:
        f.write('backbone = %s\n' %backbone)
        f.write('input_shape = %s\n' %input_shape)
        f.write('Epoch = %s\n' %Epoch)
        f.write('batch_size = %s\n' %batch_size)
        f.write('init_lr = %s\n' %init_lr)
    #---------------------------------------------------------------#
    loss_history = LossHistory(backbone, model_save_dir)
    # ------------------------------------------------#
    # ------------------------------------------------#
    #   read the corresponding txt
    # ------------------------------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    # val_lines = train_lines[:int(len(train_lines)*0.2)] # 20%
    # train_lines = train_lines[int(len(train_lines)*0.2):] # 80%
    num_train = len(train_lines)
    num_val = len(val_lines)

    #------------------------------------------------#
    #   display parameters
    #------------------------------------------------#
    show_config(
        num_classes =num_classes, backbone = backbone, input_shape = input_shape, \
        Epoch = Epoch, batch_size = batch_size, init_lr = init_lr,\
        save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    # ------------------------------------------------#
    #   adaptively adjust the learning rate according to the batchsize
    # ------------------------------------------------#
    nbs = 256
    lr_limit_max = 1e-2  # default 1e-2
    lr_limit_min = 1e-4  # default 1e-4
    Init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    # Adam
    # optimizer = optim.Adam(model.parameters(), init_lr)
    # SGD
    optimizer = optim.SGD(model.parameters(), init_lr, momentum=0.9)
    cudnn.benchmark = True
    # ------------------------------------------------#
    #   obtain the formula of learning rate decline------cosine annealing,warm up
    # ------------------------------------------------#
    lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Min_lr_fit, Epoch)
    # lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Init_lr_fit, Epoch)
    # ------------------------------------------------#
    #   dataload
    # ------------------------------------------------#
    train_dataset = DataGenerator(train_lines, input_shape=input_shape, train=True)
    val_dataset = DataGenerator(val_lines, input_shape=input_shape, train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True)

    for epoch in range(Epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(device, model, loss_history, optimizer, epoch, gen, gen_val, Epoch, model_save_dir, backbone)