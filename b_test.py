from pandas import DataFrame
import os
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.utils_metrics import plot_matrix
from utils.utils import get_im,generate_batch
import utils.recompose as rp
from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50
from nets.googlenet import googlenet
from nets.vgg import vgg16, vgg16_bn
from nets.vit import vit_b_16
from nets.swin_transformer import swin_transformer_small, swin_transformer_tiny
# from utils import filter_process
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

    # ------------------------------------------------#
    device = torch.device('cuda:0')
    # ------------------------------------------------#
    # ------------------------------------------------#


    num_workers = 8

    #   获取classes
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
    model.load_state_dict(torch.load('./logs/{}/{}_best_epoch_weights.pth'.format(backbone,backbone)))
    # Load the training images and mask
    block_image = './blocks/images'

    for ic in os.listdir(block_image):
        im_path = os.path.join(block_image,ic)
        X_test,w_n,h_n = get_im(im_path)
        # X_test = block_DataGenerator("./blocks/images/1.png",(block_height,block_weight),False)
        patches_imgs_test, extended_height, extended_width = rp.get_data_testing_overlap(
            X_test,  # original
            n_test_images=1,
            patch_height=128,
            patch_width=128,
            stride_height=128,
            stride_width=128,
            channel=3)

        mask_pred = None
        batchsize = 64
        images = generate_batch(patches_imgs_test, batchsize)

        color = [(0,255,255), (255,0, 0), (0, 0, 255), (255, 0, 255), (255, 255,0), (0, 255, 0),(0,0,0)]
        std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        with torch.no_grad():
            lis=[]
            for i, inputs in tqdm(enumerate(images)):
                inputs = torch.from_numpy(inputs).float().to(device)
                model.eval()
                outputs = model(inputs)
                preds = outputs.argmax(1)
                lis+=preds.tolist()
        for i in range(X_test.shape[0]):
            im = X_test[i]
            im = im * (std * 255) + (mean * 255)
            im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
            mask = np.zeros_like(im)

            #
            for j in range(len(lis)):
                cv2.rectangle(mask, (128*(j%w_n), 128*(j//w_n)), (128*(j%w_n+1), 128*(j//w_n+1)), color[lis[j]], -1)
            # im = np.transpose(im, (1, 2, 0))

            output = cv2.addWeighted(im, 0.5, mask, 0.5, 0)
            cv2.rectangle(output, (200, 200), (1500, 1500),
                          (255,255,255), -1)
            for ind, l in enumerate(class_names):
                cv2.putText(output, l, (500, 100 + 200 * (ind + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                                    4, color[ind], 8)


            # output=Image.fromarray(output.astype(np.uint8))
            # output.save('./blocks/preds/1.png')
            cv2.imwrite('./blocks/preds/{}'.format(ic),output)

    # result = './blocks/preds/'
    # for i in range(pred_img.shape[0]):
    #     image = np.transpose(pred_img[i, :, :, :], (1, 2, 0))
    #     for ind, l in enumerate(class_names):
    #         cv2.putText(image, l, (1360, 660 + 200 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
    #                     4, color[ind], 8)
    #     rp.visualize(image, result + im_names[i])





