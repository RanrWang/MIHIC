import numpy as np
from PIL import Image
import torch
from utils.utils import show_config,get_classes,preprocess_input

from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50
from nets.vgg import vgg16, vgg16_bn
from nets.vit import vit_b_16
from nets.googlenet import googlenet
from nets.swin_transformer import swin_transformer_small, swin_transformer_tiny
get_model_from_name = {
    "mobilenetv2"               : mobilenetv2,
    "resnet50"                  : resnet50,
    "vgg16"                     : vgg16,
    'googlenet'                 : googlenet,
    "vgg16_bn"                  : vgg16_bn,
    "vit_b_16"                  : vit_b_16,
    "swin_transformer_small"    : swin_transformer_small,
    "swin_transformer_tiny"     : swin_transformer_tiny,
}


def get_model(backbone,num_classes,model_path):
    model  = get_model_from_name[backbone](num_classes = num_classes)
    model.load_state_dict(torch.load(model_path))
    model  = model.eval().to('cuda:0')
    print('\033[1;33;44m{}\033[0m model, and classes loaded......'.format(model_path))
    return model

class Classification(object):
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   获得种类
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()
        
        show_config(**self._defaults)
    

    def generate(self):
        #---------------------------------------------------#
        #   载入模型与权值
        #---------------------------------------------------#
        if self.backbone != "vit":
            self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes)
        else:
            self.model  = get_model_from_name[self.backbone](input_shape = self.input_shape, num_classes = self.num_classes)

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model  = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))


    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image       = image.resize(self.input_shape)
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        #---------------------------------------------------------#
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(image_data)
            photo = photo.cuda(0)

            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        return preds
