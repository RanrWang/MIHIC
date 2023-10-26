import torch
import torch.nn as nn
from nets.attention_module import senet, cbam, ecanet

# VGG通用模板，只需要给定特征提取层和类别数即可
# 特征提取层：提取图像特征 avgpool：汇聚全局信息 classifier：全连接层分类器
class VGG(nn.Module):
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.cbam = cbam(512)
    # 预训练权重最后一层不匹配，所以需要初始化
        # self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.cbam(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

# 构造特征提取层的 Sequential, vgg11,13,16仅在这部分不同
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for c in cfg:
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)

# A:vgg11  B:vgg13  D:vgg16
# vgg16:224,224,3 -> 224,224,64 -> 112,112,64 -> 112,112,128 -> 56,56,128 -> 56,56,256 -> 28,28,256 -> 28,28,512
# 14,14,512 -> 14,14,512 -> 7,7,512
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

def vgg11(num_classes):
    model = VGG(make_layers(cfgs['A']), num_classes=num_classes)
    return model

def vgg13(num_classes):
    model = VGG(make_layers(cfgs['B']), num_classes=num_classes)
    return model

def vgg16(num_classes):
    model = VGG(make_layers(cfgs['D']), num_classes=num_classes)
    return model

def vgg11_bn(num_classes):
    model = VGG(make_layers(cfgs['A'], batch_norm=True), num_classes=num_classes)
    return model

def vgg13_bn(num_classes):
    model = VGG(make_layers(cfgs['B'], batch_norm=True), num_classes=num_classes)
    return model

def vgg16_bn(num_classes):
    model = VGG(make_layers(cfgs['D'], batch_norm=True), num_classes=num_classes)
    return model