import math
import random
import numpy as np
from functools import partial
from PIL import ImageFilter, ImageOps
import cv2
import os

#  获取类别名称和类别数
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# 打印所有参数
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for k, v in kwargs.items():
        print('|%25s | %40s|' % (str(k), str(v)))
    print('-' * 70)

# 余弦退火算法，学习率cos方式下降
def get_lr_scheduler(lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
    warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
    no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
    func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)


    return func

#----------------------------------------#
#   设置学习率
#----------------------------------------#
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(x):
    x /= 255
    x -= np.array([0.485, 0.456, 0.406])
    x /= np.array([0.229, 0.224, 0.225])
    return x

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
def normalize_image(image, mean, std):
    # 转换为浮点型数据类型
    image = image.astype(np.float32)

    # 缩放像素值到 0 到 1 之间
    image /= 255.0

    # 逐通道进行标准化
    for i in range(3):  # 假设是RGB图像，有三个通道
        image[:, :, i] -= mean[i]
        image[:, :, i] /= std[i]

    return image
def get_images(parent_dir):

    tissue_dir = parent_dir
    im_names = os.listdir(tissue_dir)
    # y = np.zeros((len(im_names), im_height, im_width, 1), dtype=np.float32)
    id = 1
    for im_name in im_names:
    # Load images
        img = im_name
        img_path = os.path.join(tissue_dir,img)
        x_img = cv2.imread(img_path)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_h, x_w = x_img.shape[0:2]
        w_n = int(x_w / 128)
        h_n = int(x_h / 128)
        x_img = x_img[:h_n * 128, :w_n * 128]
        X = np.zeros((1, x_img.shape[0], x_img.shape[1], 3), dtype=np.float32)
        normalized_image = normalize_image(x_img,(0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        X[id-1] = normalized_image
        id += 1
        # y[n] = mask/255.0
    return X,im_names

def get_im(img_path):
    # y = np.zeros((len(im_names), im_height, im_width, 1), dtype=np.float32)
    id = 1
    x_img = cv2.imread(img_path)
    x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
    x_h,x_w =x_img.shape[0:2]
    w_n = int(x_w/128)
    h_n = int(x_h/128)
    x_img = x_img[:h_n*128,:w_n*128]
    X = np.zeros((1, x_img.shape[0], x_img.shape[1], 3), dtype=np.float32)
    normalized_image = normalize_image(x_img,(0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    X[id-1] = normalized_image
    id += 1
        # y[n] = mask/255.0
    return X,w_n,h_n

def generate_batch(patches_imgs_test,batchsize):
    image = None
    images = np.zeros((patches_imgs_test.shape[0] // batchsize + 1, batchsize, 3, 128, 128))
    i = 0
    while i < patches_imgs_test.shape[0]:
        d = i // batchsize
        for j in range(batchsize):
            if j == 0:
                image = patches_imgs_test[j + d * batchsize, :, :, :]
                image = image[np.newaxis, :]
            else:
                if j + d * batchsize < patches_imgs_test.shape[0]:
                    data = patches_imgs_test[j + d * batchsize, :, :, :]
                    data = data[np.newaxis, :]
                else:
                    data = np.zeros((1, 3, 128, 128))
                image = np.concatenate([image, data])
        images[d, :, :, :, :] = image
        i += batchsize
    return images