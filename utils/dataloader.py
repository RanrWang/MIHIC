from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.utils import GaussianBlur

class DataGenerator(Dataset):
    def __init__(self, annotation_lines,input_shape,train=True):
        self.annotation_lines   = annotation_lines
        if train:
            self.transform = transforms.Compose([
                transforms.Resize(input_shape),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                GaussianBlur(0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(input_shape),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
    def __len__(self):
        return len(self.annotation_lines)
    
    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split('\n')[0]
        tag = torch.tensor(int(self.annotation_lines[index].split(';')[0])).long()

        image = Image.open(annotation_path)
        image = self.transform(image)

        return image, tag

# import numpy as np
# import cv2
# class block_DataGenerator(Dataset):
#     def __init__(self, im):
#         self.im = im
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])
#
#     def __len__(self):
#         return len(self.im)
#     def __getitem__(self,index):
#         # im = np.transpose()
#         im = self.im[index]
#         image = np.transpose(im,(1,2,0))
#         image = self.transform(image)
#
#
#         return image
