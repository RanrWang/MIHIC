import math
import torch
import torch.nn as nn

# 通道注意力
class senet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(senet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y

class channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.fc(self.avgpool(x).view(b,c))
        max_out = self.fc(self.maxpool(x).view(b,c))
        out     = self.sigmoid(avg_out + max_out).view(b,c,1,1)
        return x * out

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out     = torch.mean(x, dim=1, keepdim=True)
        max_out, _  = torch.max(x, dim=1, keepdim=True)
        out         = torch.cat([avg_out,max_out],dim=1)
        out         = self.sigmoid(self.conv(out))
        return x * out

class cbam(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = channel_attention(channel=channel,ratio=ratio)
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ecanet(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        out = self.avgpool(x).view(b,1,c)
        out = self.sigmoid(self.conv(out)).view(b, c, 1, 1)
        return x * out
        
        
if __name__ == "__main__":
    input = torch.ones([2,512,26,26])
    model = ecanet(512)
    print(model)
    out = model(input)
    print(out)