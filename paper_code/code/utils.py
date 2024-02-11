# -*- coding: utf-8 -*-
# @Time     : 2022/12/30 10:37
# @Function : 
# @File     : utils.py

import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torch_nets import (
    tf2torch_inception_v3,
    tf2torch_inception_v4,
    tf2torch_resnet_v2_101,
    tf2torch_inc_res_v2,
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)
# from torch_nets.FD_ResNet import resnet101_denoise, resnet152_denoise, get_normalize_layer
# from pytorch_pretrained_vit import ViT
# from torchvision import models
# from torch_nets.Sequencer_Deep_LSTM import get_sequencer_deep_lstm
# from torch_nets.NRP_networks import *
# from torch_nets.NRP_utils import *


def tensor2im(input_image, mean, std, normalize=False, imtype=np.uint8):
    # mean = [0.485,0.456,0.406] 
    # std = [0.229,0.224,0.225]
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if normalize:
            for i in range(len(mean)):
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        # image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


class Normalize(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        (input - mean) / std
        ImageNet normalize:
            'tensorflow': mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            'torch': mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()

        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


def get_model(net_name, model_dir):
    """Load converted model"""
    if "tf2torch" in net_name:
        model_path = os.path.join(model_dir, net_name + '.npy')
        if net_name == 'tf2torch_inception_v3':
            net = tf2torch_inception_v3
        elif net_name == 'tf2torch_inception_v4':
            net = tf2torch_inception_v4
        # elif net_name == 'tf2torch_resnet_v2_50':
        #     net = tf2torch_resnet_v2_50
        elif net_name == 'tf2torch_resnet_v2_101':
            net = tf2torch_resnet_v2_101
        # elif net_name == 'tf2torch_resnet_v2_152':
        #     net = tf2torch_resnet_v2_152
        elif net_name == 'tf2torch_inc_res_v2':
            net = tf2torch_inc_res_v2
        elif net_name == 'tf2torch_adv_inception_v3':
            net = tf2torch_adv_inception_v3
        elif net_name == 'tf2torch_ens3_adv_inc_v3':
            net = tf2torch_ens3_adv_inc_v3
        elif net_name == 'tf2torch_ens4_adv_inc_v3':
            net = tf2torch_ens4_adv_inc_v3
        elif net_name == 'tf2torch_ens_adv_inc_res_v2':
            net = tf2torch_ens_adv_inc_res_v2
        else:
            print('Wrong model name:', net_name, '!')
            exit()

        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            net.KitModel(model_path).eval(), )
    else:
        print('Wrong model name:', net_name, '!')
        exit()
    #
    # else:
    #     if net_name == 'resnet101_denoise':
    #         model = resnet101_denoise()
    #         model.load_state_dict(torch.load("./models/Adv_Denoise_Resnext101.pytorch"))
    #         model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    #         model = nn.Sequential(get_normalize_layer(), model)
    #     elif net_name == 'resnet152_denoise':
    #         model = resnet152_denoise()
    #         model.load_state_dict(torch.load("./models/Adv_Denoise_Resnet152.pytorch"))
    #         model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    #         model = torch.nn.Sequential(get_normalize_layer(), model)
    #     elif net_name == 'NRP_ens3_adv_inc_v3':
    #         netG = NRP(3, 3, 64, 23)
    #         netG.load_state_dict(torch.load('models/NRP.pth'))
    #         model_path = os.path.join(model_dir, 'tf2torch_ens3_adv_inc_v3.npy')
    #         net = tf2torch_ens3_adv_inc_v3
    #         model = nn.Sequential(
    #             netG,
    #             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #             net.KitModel(model_path).eval(), )
    #     elif net_name == 'NRP_resG_ens3_adv_inc_v3':
    #         netG = NRP_resG(3, 3, 64, 23)
    #         netG.load_state_dict(torch.load('models/NRP_resG.pth'))
    #         model_path = os.path.join(model_dir, 'tf2torch_ens3_adv_inc_v3.npy')
    #         net = tf2torch_ens3_adv_inc_v3
    #         model = nn.Sequential(
    #             netG,
    #             Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #             net.KitModel(model_path).eval(), )
    #     elif net_name == 'ViT-B/16':
    #         model = ViT('B_16_imagenet1k', pretrained=True)
    #         model = nn.Sequential(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                               model.eval())
    #     elif net_name == 'NASNet-L':
    #         model = models.mnasnet1_0(pretrained=True)
    #         model = nn.Sequential(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                               model.eval())
    #     elif net_name == 'Sequencer_Deep_LSTM':
    #         model = get_sequencer_deep_lstm(model_dir)
    #         model = nn.Sequential(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #                               model.eval())

    return model


class ImageNet(Dataset):
    """load data from img and csv"""
    def __init__(self, dir, csv_path, transforms=None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel']
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, Truelabel

    def __len__(self):
        return len(self.csv)

if __name__ == '__main__':
    a = get_model('Sequencer_Deep_LSTM', 'models')
    print(a)

