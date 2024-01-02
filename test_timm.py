from pandas import DataFrame
import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.utils_metrics import plot_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from utils.dataloader import DataGenerator
from utils.utils import get_classes, show_config, get_lr_scheduler, set_optimizer_lr

if __name__ == "__main__":
    # ------------------------------------------------#
    #   load the model from timm
    from timm.models import swin_tiny_patch4_window7_224, coat_tiny
    from timm.models.convnext import convnext_tiny
    from timm.models import convit_tiny, crossvit_tiny_240
    from timm.models.efficientnet import efficientnet_b0
    from timm.models.hrnet import hrnet_w18
    from timm.models.cait import cait_s24_224
    from timm.models.twins import twins_pcpvt_base, twins_svt_base
    # -------------------------------------------------#
    '''
    # coat_tiny: coat_tiny. batchsize=512. img_size is set as (128,128).not processing resize operation. However is can be resize to(224,224).
    # swin_timm_tiny: swin_tiny_patch4_window7_224. batchsize=256. For match the window_size=7, img_size is set as (224,224). 
    The other version(swin_tiny) perform '_make_divisible function', which is used to ensure that a given value 
    is divisible by a specified divisor, so img_size in the other version(swin_tiny) is (128,128), window_size=7.
    # convnext_tiny: convnext_tiny. batchsize=512. img_size=128.
    # convit_tiny: convit_tiny. batchsize=512. img_size is resize to (224,224). The other parameter is default.
    # efficientnet_b0: efficientnet_b0. batchsize=512. img_size is (128,128). But it also can be resize to (224,224).
    # hrnet_w18: hrnet_w18. batchsize=256. img_size=128. But it can also be resize to (224,224).
    # cait: cait_s24_224. batchsize=128. img_size=224. The training time of this model is too long, so it has not been trained.
    # crossvit_tiny: crossvit_tiny_240. img_size=128. 
    # twins_pcpvt_base: twins_papvt_base. batchsize=512. img_size can be as (128,128) or (224,224).
    # twins_svt_base: twins_svt_base. batchsize=512. img_size can be set as (128,128) or (224,224).
    '''
    # backbone = 'coat_tiny'
    # backbone = 'swin_timm_tiny'
    backbone = 'convnext_tiny'
    # backbone = 'convit_tiny'
    # backbone = 'efficientnet_b0'
    # backbone = 'hrnet_w18'
    # backbone = 'cait'
    # backbone ='crossvit_tiny'
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
        input_shape = [224,224]
    if backbone == 'crossvit_tiny':
        model = crossvit_tiny_240(pretrained=True, num_classes=7)
        # input_shape = [224,224]
    if backbone == 'twins_pcpvt_base':
        model = twins_pcpvt_base(pretrained=True, num_classes=7)
        # input_shape = [224,224]
    if backbone == 'twins_svt_base':
        model = twins_svt_base(pretrained=True, num_classes=7)
        # input_shape = [224,224]
    # ----------------------------------------------------------------#
    # ------------------------------------------------#
    #   class name txt
    # ------------------------------------------------#
    classes_path = 'model_data/cls_names.txt'
    # ------------------------------------------------#
    #   obtain the classes
    # ------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    # ------------------------------------------------#
    #   define the image_size
    # ------------------------------------------------#
    input_shape = [128, 128]
    # ------------------------------------------------#
    #   usage of the GPU
    # ------------------------------------------------#
    device = torch.device('cuda:0')
    # ------------------------------------------------#
    # ------------------------------------------------#
    #   train_annotation_path   training images path and labels
    # ------------------------------------------------#
    test_annotation_path = './dataset_path/test.txt'
    weight_path = './logs/' + backbone
    # ------------------------------------------------#
    batch_size = 256
    num_workers = 8
    # ------------------------------------------------#
    #   model initialization
    # ------------------------------------------------#
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load('{}/{}_best_epoch_weights.pth'.format(weight_path,backbone)))

    # ------------------------------------------------#
    #   read the corresponding txt.file
    # ------------------------------------------------#
    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines = f.readlines()

    random.shuffle(test_lines)
    test_lines = test_lines[:]
    num_test = len(test_lines)

    # ------------------------------------------------#
    #   dataload
    # ------------------------------------------------#
    test_dataset = DataGenerator(test_lines, input_shape=input_shape, train=False)
    gen_test = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=False)
    model.to(device)
    model.eval()
    running_corrects = 0
    xlsdata = {
        'groundtruth': [],
        'prediction': []
    }
    groundtruth = []
    prediction = []
    with torch.no_grad():
        for data in gen_test:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            groundtruth += labels.tolist()
            model.eval()
            outputs = model(inputs)
            preds = outputs.argmax(1)
            # output_numpy = outputs.cpu().detach().numpy()
            # labels_numpy = labels.cpu().detach().numpy()
            prediction += preds.tolist()
            running_corrects += torch.sum(preds == labels.data)
            for i in range(inputs.size(0)):
                xlsdata['groundtruth'].append(class_names[labels[i]])
                xlsdata['prediction'].append(class_names[preds[i]])

    acc = float(running_corrects) / len(test_dataset)
    recall =recall_score(groundtruth,prediction,average=None)
    precision = precision_score(groundtruth,prediction,average=None)
    f1 = f1_score(groundtruth,prediction,average=None)
    xlsdata['groundtruth'].append('acc')
    xlsdata['prediction'].append(acc)
    xlsdata['groundtruth'].append('recall')
    xlsdata['prediction'].append(recall)
    xlsdata['groundtruth'].append('precision')
    xlsdata['prediction'].append(precision)
    xlsdata['groundtruth'].append('F1-Score')
    xlsdata['prediction'].append(f1)
    print('recall:', recall)
    print('precision:', precision)
    print('F1-Score:', f1)
    print('acc:', acc)
    df = DataFrame(xlsdata)
    df.to_excel('./evaluation/acc/{}_acc.xlsx'.format(backbone))
    title=backbone+'-confusion_matrix'
    plot_matrix(groundtruth, prediction,backbone, [0, 1, 2, 3, 4, 5, 6], title=title,
                axis_labels=class_names)
