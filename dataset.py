import torch
import torch.utils.data as data
from torchvision import transforms


import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class FishDataset(data.Dataset):
    def __init__(self, data_path, filenames, is_train = False):
        super(FishDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.interp = Image.ANTIALIAS
        self.is_train = is_train
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

    def preprocess(self, img, color_aug):
        img_aug = self.to_tensor(color_aug(img))
        return img_aug
    
    def generate_gt(self, folder):
        gt_list = ["saithe", "herring", "grey_gurnard", "norway_pout", "anchovy",
                    "red_mullet", "cod", "haddock", "sardine", "mackerel"]
        #gt = torch.from_numpy(gt_list.index(folder))
        gt = gt_list.index(folder)
        return gt
    
    def __getitem__(self, index):
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        img_name = line[1]
        gt = self.generate_gt(folder)
        color = self.loader(os.path.join(self.data_path, folder, img_name))
        
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT) 
        img = color 
        
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        img_aug = self.preprocess(img, color_aug)
        del img 
        return img_aug, gt 
    def __len__(self):
        return len(self.filenames)

