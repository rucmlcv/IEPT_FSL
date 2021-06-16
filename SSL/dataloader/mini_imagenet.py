import os.path as osp
import PIL
from PIL import Image
import torchvision.transforms.functional as TF

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/miniImagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniImagenet/split')

class MiniImageNet(Dataset):

    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.args = args
        
        if args.model_type == 'ConvNet':
            # for ConvNet512 and Convnet64
            image_size = 84
            self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                                                    np.array([0.229, 0.224, 0.225]))
                                                ])
            
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size)
            ])            
        else:
            # for Resnet12
            image_size = 84
                     
            self.to_tensor = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
                                                ])
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size)
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
                
        image_0 = self.to_tensor(image)
        image_90 = self.to_tensor(TF.rotate(image, 90))
        image_180 = self.to_tensor(TF.rotate(image, 180))
        image_270 = self.to_tensor(TF.rotate(image, 270))

        all_images = torch.stack([image_0, image_90, image_180, image_270], 0) # <4, 3, size, size>
        
        return all_images, label



       
