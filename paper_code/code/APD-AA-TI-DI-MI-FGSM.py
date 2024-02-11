# -*- coding: utf-8 -*-
# @Time     : 2022/12/25 16:56
# @Function : Implement of APD-AA-TI-DI-MI-FGSM
# @File     : APD-AA-TI-DI-MI-FGSM.py


from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms, models, utils
from torchvision.transforms.functional import to_pil_image
import numpy as np
from tqdm import tqdm
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
import sys
import os
from PIL import Image
from utils import tensor2im
from utils import get_model, ImageNet
import seaborn
from torchcam.methods import SmoothGradCAMpp, GradCAMpp, GradCAM
from torchcam.utils import overlay_mask
import scipy.stats as st
print("start time is : %s" % datetime.now())

# Parameters
Dropout = True  # on-off of our APD method
batch_size = 1
epsilon = 16/255
lr = 1.6
iter = 10
is_momentum = True
momentum = 1
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Generate Adversarial Examples
def fgsm_attack(image, clean_image, epsilon, lr, data_grad):
    data_grad = data_grad.sign()

    data_max = clean_image + epsilon
    data_min = clean_image - epsilon
    data_max.clamp_(0, 1)
    data_min.clamp_(0, 1)

    with torch.no_grad():
        sign_data_grad = data_grad.sign()
        perturbed_image = image + lr * sign_data_grad
        perturbed_image.data = torch.max(torch.min(perturbed_image.data, data_max), data_min)

    return perturbed_image

# TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

# DI
def input_diversity(input_tensor):
    rnd = torch.randint(299, 330, ())
    rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=True)
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [330, 330])
    return padded if torch.rand(()) < 0.5 else input_tensor

# Load ImageNet
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])
data_dir = './imagenet-val/image_val_1000/images'
csv_path = './imagenet-val/image_val_1000/dev_dataset.csv'
test_dataset = ImageNet(data_dir, csv_path, transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# load model
list_nets = [
    'tf2torch_inception_v3',
    'tf2torch_inception_v4',
    'tf2torch_inc_res_v2',
    'tf2torch_resnet_v2_101',
    'tf2torch_ens3_adv_inc_v3',
    'tf2torch_ens4_adv_inc_v3',
    'tf2torch_ens_adv_inc_res_v2'
]

source_model = 0
net = get_model(list_nets[source_model], "./models/")
net.to(device)
net.eval()

target_nets = []
for i, net_name in enumerate(list_nets):
    target_nets.append(get_model(net_name, "./models/"))
    target_nets[i].to(device)
    target_nets[i].eval()


# === RUN ===
cam_extractor = GradCAMpp(net, input_shape=(3, 299, 299))
correct_num = [0 for _ in range(len(list_nets))]
total_num = [0 for _ in range(len(list_nets))]
for i, (data, target) in enumerate(test_loader):

    # =================== Basic Setting =================
    data, target = data.to(device), target.to(device)
    clean_data = data.clone().detach()
    data.requires_grad = True
    # ==================================================

    # ==================================================
    # ===================== Update =====================
    # ==================================================
    g = torch.zeros(data.size(0), 1, 1, 1).cuda()
    for j in range(iter):
        AA_backup_data = data.clone().detach()
        for AA_copy_num in range(3):

            eta = 0.2
            Admix_data = AA_backup_data.clone().detach()
            for ad_num in range(Admix_data.size(0)):
                Admix_data[ad_num] = Admix_data[ad_num] + eta*AA_backup_data[(ad_num+AA_copy_num+1)%Admix_data.size(0)].clone().detach()

            for SI_copy_num in range(5):
                data = Admix_data.clone().detach() / (2**SI_copy_num)
                data.requires_grad = True

                if Dropout:
                    backup_data = data.clone().detach()
                    # =================== Get CAM ===================
                    data.detach_()
                    activation_values = []
                    for image_num in range(data.size(0)):
                        output = net(data[image_num].unsqueeze(0))
                        activation_map = cam_extractor(target[image_num].squeeze(0).argmax().item(), output)
                        net.zero_grad()
                        activation_value = np.array(
                            to_pil_image(activation_map[0].squeeze(0), mode='F').resize(to_pil_image(
                                torch.from_numpy(
                                    tensor2im(clean_data[image_num].unsqueeze(0)[0].detach().cpu(), mean, std))).size,
                                                                                        resample=Image.BICUBIC))
                        activation_values.append(activation_value)
                    activation_values = np.array(activation_values)
                    # =================== Calculate the local maximum ===================
                    coordinates_for_maps = []
                    for activation_value in activation_values:
                        coordinates_for_a_map = []
                        bigger_than_3 = False
                        for w in range(1, activation_value.shape[0] - 1):
                            for h in range(1, activation_value.shape[1] - 1):
                                temp_data = []
                                temp_data.append(activation_value[w - 1][h])
                                temp_data.append(activation_value[w + 1][h])
                                temp_data.append(activation_value[w][h - 1])
                                temp_data.append(activation_value[w][h + 1])
                                if activation_value[w][h] > max(temp_data):
                                    coordinates_for_a_map.append([w, h])
                                if len(coordinates_for_a_map) >= 3:
                                    bigger_than_3 = True
                                    break
                            if bigger_than_3:
                                break
                        coordinates_for_maps.append(coordinates_for_a_map)
                     # =================== Calculate Gradient ===================
                    data_grad = []
                    aaa = [15, 30, 45, 60, 75]
                    nums_images = len(aaa)
                    for image_num in range(data.size(0)):
                        coordinates = coordinates_for_maps[image_num]
                        if len(coordinates) == 0:
                            data_batch = data[image_num, :, :, :].clone().detach().unsqueeze(0)
                            target_batch = target[image_num].unsqueeze(0)
                        else:
                            data_batch = torch.stack(
                                [data[image_num, :, :, :].clone().detach() for _ in coordinates] * nums_images, 0)
                            target_batch = torch.stack([target[image_num] for _ in coordinates] * nums_images, 0)
                            for coor_id, coordinate in enumerate(coordinates):
                                w, h = coordinate
                                for len_id in range(nums_images):
                                    mask_length = aaa[len_id]
                                    a = max(w - mask_length, 0)
                                    b = min(w + mask_length, 299)
                                    c = max(h - mask_length, 0)
                                    d = min(h + mask_length, 299)
                                    data_batch[coor_id * nums_images + len_id, :, a:b, c:d] = clean_data[image_num, :,
                                                                                              a:b,
                                                                                              c:d].clone()
                        data_batch.requires_grad = True
                        output = net(input_diversity(data_batch))
                        loss = F.cross_entropy(output, target_batch)
                        net.zero_grad()
                        loss.backward()
                        temp_data_grad = torch.mean(data_batch.grad.data, dim=0)
                        data_grad.append(temp_data_grad)
                    data_grad = torch.stack(data_grad, dim=0)
                else:
                    output = net(input_diversity(data))
                    loss = F.cross_entropy(output, target)
                    net.zero_grad()
                    loss.backward()
                    data_grad = data.grad.clone().to(device)
                    data_grad = F.conv2d(data_grad, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
                    data_grad = data_grad.data

                if SI_copy_num == 0:
                    SI_data_grad = data_grad
                else:
                    SI_data_grad += data_grad
            data_grad = SI_data_grad / 5

            if AA_copy_num == 0:
                AA_data_grad = data_grad
            else:
                AA_data_grad += data_grad
        data_grad = AA_data_grad / 5
        data = AA_backup_data.clone().detach()

        # ==============================================

        # =================== Update =====================
        if is_momentum:
            g = momentum * g.data + data_grad / torch.mean(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)
            data_grad = g.clone().detach()
        data = fgsm_attack(data, clean_data, epsilon, lr, data_grad)
        data.detach_()
        data.requires_grad = True
        # ==============================================

    # ==================================================
    # ==================================================

    # ==================== Attack =========================
    with torch.no_grad():
        if i % 5 == 0:
            print("="*50)
            print("now time is : %s" % datetime.now())
        for net_num, net2 in enumerate(target_nets):
            pred = net2(data).max(1)[1]
            correct_num[net_num] += int(torch.sum(target != pred))
            total_num[net_num] += len(target)
            if i % 5 == 0:
                print("{:4} {:30}: Attack Success Rate: {:.2f}%".format(i, list_nets[net_num],
                                                                        correct_num[net_num]/total_num[net_num]*100))
    # ==================================================


# ================================= Results =======================================
print("================================= Results =======================================")
print("finish time is : %s" % datetime.now())
print("Source Model", list_nets[source_model])
for net_num in range(len(list_nets)):
    print("{:30}: Attack Success Rate: {:.2f}%".format(list_nets[net_num], correct_num[net_num]/total_num[net_num]*100))
print("=================================================================================")

