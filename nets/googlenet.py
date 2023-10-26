# import torch
# import torch.nn as nn

# class Inception(nn.Module):
#     def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
#         super().__init__()

#         #1x1conv branch
#         self.b1 = nn.Sequential(
#             nn.Conv2d(input_channels, n1x1, kernel_size=1),
#             nn.BatchNorm2d(n1x1),
#             nn.ReLU(inplace=True)
#         )

#         #1x1conv -> 3x3conv branch
#         self.b2 = nn.Sequential(
#             nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
#             nn.BatchNorm2d(n3x3_reduce),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
#             nn.BatchNorm2d(n3x3),
#             nn.ReLU(inplace=True)
#         )

#         #1x1conv -> 5x5conv branch
#         #we use 2 3x3 conv filters stacked instead
#         #of 1 5x5 filters to obtain the same receptive
#         #field with fewer parameters
#         self.b3 = nn.Sequential(
#             nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
#             nn.BatchNorm2d(n5x5_reduce),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
#             nn.BatchNorm2d(n5x5, n5x5),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
#             nn.BatchNorm2d(n5x5),
#             nn.ReLU(inplace=True)
#         )

#         #3x3pooling -> 1x1conv
#         #same conv
#         self.b4 = nn.Sequential(
#             nn.MaxPool2d(3, stride=1, padding=1),
#             nn.Conv2d(input_channels, pool_proj, kernel_size=1),
#             nn.BatchNorm2d(pool_proj),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


# class GoogleNet(nn.Module):

#     def __init__(self, num_classes=100):
#         super().__init__()
#         self.prelayer = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(192),
#             nn.ReLU(inplace=True),
#         )

#         #although we only use 1 conv layer as prelayer,
#         #we still use name a3, b3.......
#         self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
#         self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

#         ##"""In general, an Inception network is a network consisting of
#         ##modules of the above type stacked upon each other, with occasional
#         ##max-pooling layers with stride 2 to halve the resolution of the
#         ##grid"""
#         self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

#         self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
#         self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
#         self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
#         self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
#         self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

#         self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
#         self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

#         #input feature size: 8*8*1024
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout2d(p=0.4)
#         self.linear = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         x = self.prelayer(x)
#         x = self.maxpool(x)
#         x = self.a3(x)
#         x = self.b3(x)

#         x = self.maxpool(x)

#         x = self.a4(x)
#         x = self.b4(x)
#         x = self.c4(x)
#         x = self.d4(x)
#         x = self.e4(x)

#         x = self.maxpool(x)

#         x = self.a5(x)
#         x = self.b5(x)

#         #"""It was found that a move from fully connected layers to
#         #average pooling improved the top-1 accuracy by about 0.6%,
#         #however the use of dropout remained essential even after
#         #removing the fully connected layers."""
#         x = self.avgpool(x)
#         x = self.dropout(x)
#         x = x.view(x.size()[0], -1)
#         x = self.linear(x)

#         return x
import torch
from torch import nn
from torch.nn import functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X):
        return self.relu(self.bn(self.conv(X)))
        
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, X):
        branch1 = self.branch1(X)
        branch2 = self.branch2(X)
        branch3 = self.branch3(X)
        branch4 = self.branch4(X)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)



class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, X):
        X = self.averagePool(X)
        X = self.conv(X)
        
        X = torch.flatten(X, start_dim=1)
        
        X = F.relu(self.fc1(X), inplace=True)
        X = F.dropout(X, 0.7, training=self.training)
        X = self.fc2(X)
        
        return X        
        



class GoogleNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=False):
        super().__init__()
        self.aux_logits = aux_logits
        
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        if self.aux_logits:
        	# 如果加入辅助分类器
            self.aux1 = AuxClassifier(512, num_classes)
            self.aux2 = AuxClassifier(528, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        for m in self.modules():
        	# 这里不显式初始化的话，pytorch也是会默认kaiming初始化，但稍有一点不同
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def forward(self, X):
        X = self.conv1(X)
        X = self.maxpool1(X)
        X = self.conv2(X)
        X = self.maxpool2(X)
        
        X = self.inception3a(X)
        X = self.inception3b(X)
        X = self.maxpool3(X)
        
        X = self.inception4a(X)
        if self.training and self.aux_logits:
            aux1 = self.aux1(X)
        X = self.inception4b(X)
        X = self.inception4c(X)
        X = self.inception4d(X)
        if self.training and self.aux_logits:
            aux2 = self.aux2(X)
        X = self.inception4e(X)
        X = self.maxpool4(X)
        
        X = self.inception5a(X)
        X = self.inception5b(X)
        
        X = self.avgpool(X)
        X = torch.flatten(X, start_dim=1)
        X = self.dropout(X)
        X = self.fc(X)
        
        if self.training and self.aux_logits:
            return X, aux2, aux1
        return X

def googlenet(num_classes):
    return GoogleNet(num_classes=num_classes)