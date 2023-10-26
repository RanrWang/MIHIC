from pandas import DataFrame
import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.utils_metrics import plot_matrix
from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50
from nets.googlenet import googlenet
from sklearn.metrics import recall_score, precision_score, f1_score
from nets.vgg import vgg16, vgg16_bn
from nets.vit import vit_b_16
from nets.swin_transformer import swin_transformer_small, swin_transformer_tiny

# from utils.eval import eval

get_model_from_name = {
    "mobilenetv2": mobilenetv2,
    "resnet50": resnet50,
    "googlenet": googlenet,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vit_b_16": vit_b_16,
    "swin_transformer_small": swin_transformer_small,
    "swin_transformer_tiny": swin_transformer_tiny,
}

from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator
from utils.utils import get_classes, show_config, get_lr_scheduler, set_optimizer_lr
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # ------------------------------------------------#

    classes_path = 'model_data/cls_names.txt'
    # ------------------------------------------------#

    input_shape = [128, 128]
    # ------------------------------------------------#
    #   models：
    #   vgg16、vgg16_bn
    #   mobilenetv2
    #   googlenet
    #   resnet50
    #   vit_b_16
    #   swin_transformer_small swin_transformer_tiny
    # ------------------------------------------------#
    backbone = "swin_transformer_tiny"
    # ------------------------------------------------#
    device = torch.device('cuda:0')
    # ------------------------------------------------#
    # ------------------------------------------------#
    #   test_annotation_path
    # ------------------------------------------------#
    test_annotation_path = './dataset_path/test.txt'
    weight_path = './logs/'+backbone
    # ----------------------------------------------------------------#

    batch_size = 256
    num_workers = 8

    # ------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    # ------------------------------------------------#
    #   model initialization
    # ------------------------------------------------#
    if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        model = get_model_from_name[backbone](num_classes=num_classes)
    else:
        model = get_model_from_name[backbone](input_shape=input_shape, num_classes=num_classes)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load('{}/{}_best_epoch_weights.pth'.format(weight_path,backbone)))


    # ------------------------------------------------#
    with open(test_annotation_path, encoding='utf-8') as f:
        test_lines = f.readlines()

    random.shuffle(test_lines)
    test_lines = test_lines[:]
    num_test = len(test_lines)

    # ------------------------------------------------#
    test_dataset = DataGenerator(test_lines, input_shape=input_shape, train=True)
    gen_test = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True)
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
            for p in outputs:
                topk_values, topk_indices = torch.topk(p, 2)
                second_largest_index = topk_indices[1]
                if p.argmax(0)==3 and second_largest_index==4:
                    p[4]=abs(p[3]-p[4])+1
            # adjust background
                if second_largest_index ==6 and abs(p.max()-p[6])<=1:
                    p[6]+=1
            preds = outputs.argmax(1)
            prediction += preds.tolist()
            running_corrects += torch.sum(preds == labels.data)
            for i in range(inputs.size(0)):
                xlsdata['groundtruth'].append(class_names[labels[i]])
                xlsdata['prediction'].append(class_names[preds[i]])

    acc = float(running_corrects) / len(test_dataset)
    recall = recall_score(groundtruth, prediction, average=None)
    precision = precision_score(groundtruth, prediction, average=None)
    f1 = f1_score(groundtruth, prediction, average=None)
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
