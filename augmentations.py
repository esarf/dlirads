from __future__ import print_function, division

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import functools
import random
import time
import kornia
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import RandomThinPlateSpline, RandomAffine, RandomCrop, RandomErasing, RandomHorizontalFlip, \
    RandomRotation, RandomGaussianNoise, RandomGaussianBlur
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from skimage import color
from skimage import io, transform
import time

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import kornia




#### SPATIAL AUGMENTATION #####

class RandomRotate_(object):
    """Input shape: (B,2,H,W)
    A grid of 4 points for each dimension.
    Output shape: (B,2,H,W)
    A transformed grid of 4 points for each dimension."""

    def __init__(self, dims='XY', angle_min=-30.,
                 angle_max=30., device='cuda', p=0.5, batch_size=64):
        self.dims = dims
        self.angle_min = angle_min * math.pi / 180.
        self.angle_max = angle_max * math.pi / 180.
        self.device = device
        self.p = p
        self.batch_size = batch_size

    def __call__(self, grids, config):
        self.probas = torch.tensor([1 - self.p, self.p], device=self.device, dtype=torch.float).multinomial(
            num_samples=self.batch_size, replacement=True).type(torch.bool)

        angle = (torch.rand((self.batch_size,), dtype=grids.dtype, device=grids.device))
        angle = angle * (self.angle_max - self.angle_min) + self.angle_min
        ### set to 0 where proba is False
        angle[~self.probas] = 0

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        rot_mat = (cos, -sin, sin, cos)
        rot_mat = torch.stack(rot_mat, -1).reshape(angle.shape + (2, 2))

        rot_grid = torch.matmul(rot_mat.to(self.device),
                                grids.reshape(self.batch_size, 2, -1))  # shape (batch_size, 2, h_dim*w_dim)

        torch.cuda.empty_cache()

        # return rot_grid.reshape(batch_size,3,D,H,W)
        return rot_grid.reshape(grids.shape)


## translation

class RandomTranslate_(object):
    """Input shape: (B,2,H,W)
     A grid of 4 points for each dimension.
     Output shape: (B,2,H,W)
     A transformed grid of 4 points for each dimension."""

    def __init__(self, max_width=100, max_height=100,
                 device='cuda', W=512, H=512, p=0.5, batch_size=64):
        self.max_width = (max_width / W) * 2
        self.max_height = (max_height / H) * 2
        self.device = device
        self.p = p
        self.batch_size = batch_size

    def __call__(self, grids, config):
        self.probas = torch.tensor([1 - self.p, self.p], device=self.device, dtype=torch.float).multinomial(
            num_samples=self.batch_size, replacement=True).type(torch.bool)

        with torch.no_grad():
            x_ = (2 * self.max_width) * torch.rand((self.batch_size,), device=self.device) - self.max_width
            y_ = (2 * self.max_height) * torch.rand((self.batch_size,), device=self.device) - self.max_height

            translation_vector = torch.stack([x_, y_], dim=-1)  # B vecteurs (x,y) shape (batch_size,2)
            flow = translation_vector.view(self.batch_size, 2, 1, 1)  # flow vector: shape (batch_size,2)

            grids[self.probas] += flow[self.probas]

            torch.cuda.empty_cache()

        return grids


## flips

## horizontal flip

class RandomHorizontalFlip_(object):
    """Input shape: (B,1,H,W)
     Output shape: (B,1,H,W)"""

    def __init__(self, device='cuda', p=0.5, batch_size=64):
        self.device = device
        self.p = p
        self.batch_size = batch_size

    def __call__(self, img, config):
        self.probas = torch.tensor([1 - self.p, self.p], device=self.device, dtype=torch.float).multinomial(
            num_samples=self.batch_size, replacement=True).type(torch.bool)

        with torch.no_grad():
            img[self.probas, :] = img[self.probas, :].data.flip(dims=[3])
        return img


## vertical flip

class RandomVerticalFlip_(object):
    """Input shape: (B,1,H,W)
     Output shape: (B,1,H,W)"""

    def __init__(self, device='cuda', p=0.5, batch_size=64):
        self.device = device
        self.p = p
        self.batch_size = batch_size

    def __call__(self, img, config):
        self.probas = torch.tensor([1 - self.p, self.p], device=self.device, dtype=torch.float).multinomial(
            num_samples=self.batch_size, replacement=True).type(torch.bool)

        with torch.no_grad():
            img[self.probas, :] = img[self.probas, :].data.flip(dims=[2])
        return img

    ## cropping


class RandomResizedCrop_(object):
    """Input shape: (B,2,H,W)
     Output shape: (B,2,H,W)"""

    def __init__(self, crop_max=0.3, device='cuda',
                 p=0.5, batch_size=64):
        self.crop_max = crop_max
        self.device = device
        self.p = p
        self.batch_size = batch_size

    def __call__(self, grids, config):
        with torch.no_grad():
            self.probas = torch.tensor([1 - self.p, self.p], device=self.device, dtype=torch.float).multinomial(
                num_samples=self.batch_size, replacement=True).type(torch.bool)

            crop = torch.rand((self.batch_size, 1), dtype=grids.dtype, device=grids.device)
            crop = (1. - self.crop_max * crop)
            crop = crop.view(-1, 1, 1, 1).expand(grids.size())
            grids[self.probas] *= crop[self.probas]

            # newgrid = torch.stack((newgrid[:,0,:,:,0],newgrid[:,1,:,:,0],newgrid[:,2,:,:,0]),dim=3)

        torch.cuda.empty_cache()

        return grids


#### IMAGE AUGMENTATION ####


## cutout

class RandomCutout_(object):

    def __init__(self, width_ratio=0.2, height_ratio=0.2,
                 device='cuda', p=0.5, batch_size=64):
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio
        self.device = device
        self.p = p
        self.batch_size = batch_size

    def __call__(self, img, config):
        with torch.no_grad():
            self.probas = torch.tensor([1 - self.p, self.p], device=self.device, dtype=torch.float).multinomial(
                num_samples=self.batch_size, replacement=True)

            W = img.shape[3]
            H = img.shape[2]
            width = int(self.width_ratio * W)
            height = int(self.height_ratio * H)

            indxs = (self.probas == 1).nonzero(as_tuple=True)[0]

            lenindxs = len(indxs)

            if lenindxs > 0:
                img_ = img[self.probas.bool()].view(lenindxs, -1)
                indices = torch.stack([torch.stack(torch.meshgrid((torch.arange(0, width, device='cuda'),
                                                                   torch.arange(0, height, device='cuda')))).view(2, width * height)]).cuda().repeat(lenindxs,1,1)

                z = torch.stack([torch.randint(0, W - width, size=(lenindxs,), device='cuda'),
                                 torch.randint(0, H - height, size=(lenindxs,), device='cuda')], dim=-1)

                indices += z.unsqueeze(2)

                index = indices[:, 0, :] * W + indices[:, 1, :]

                img_.scatter_(dim=1, index=index.cuda(), value=0)

                img[self.probas.bool()] = img_.view(lenindxs, config.input_dim, W, H)

                # clear the GPU memory
                torch.cuda.empty_cache()

        return img


## noise

class RandomGaussianNoise_(object):
    """Input shape: (B,1,H,W)
     Output shape: (B,1,H,W)"""

    def __init__(self, mean=0.0, std=0.1, device='cuda',
                 p=0.5, batch_size=64):
        self.mean = mean
        self.std = std
        self.device = device
        self.p = p
        self.batch_size = batch_size

    def __call__(self, img, config):
        self.probas = torch.tensor([1 - self.p, self.p], device=self.device, dtype=torch.float).multinomial(
            num_samples=self.batch_size, replacement=True).type(torch.bool)

        with torch.no_grad():
            W = img.shape[3]
            H = img.shape[2]
            random_noise = torch.add(torch.mul(torch.randn(self.batch_size, 1, H, W, device=img.device), self.std),
                                     self.mean).repeat(1, config.input_dim, 1, 1)
            img[self.probas] += random_noise[self.probas]

        torch.cuda.empty_cache()
        return img


## blur

class RandomGaussianBlur_(object):
    """Input shape: (B,1,H,W)
     Output shape: (B,1,H,W)"""

    def __init__(self, kernel_size=5, sigma=1, device='cuda',
                 p=0.5, batch_size=64):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device
        self.p = p
        self.batch_size = batch_size

        with torch.no_grad():
            # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
            xy_grid = torch.stack(torch.meshgrid(torch.arange(self.kernel_size, device='cuda'),
                                                 torch.arange(self.kernel_size, device='cuda'),
                                                 indexing='xy'),dim=-1)

            mean = (self.kernel_size - 1) / 2.
            variance = self.sigma ** 2.

            # Calculate the 2-dimensional gaussian kernel which is
            # the product of two gaussian distributions for two different
            # variables (in this case called x and y)
            gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
                -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))  # shape (kernel_size,kernel_size)
            # Make sure sum of values in gaussian kernel equals 1.
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
            gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

            self.gaussian_filter = torch.nn.Conv2d(in_channels=3, out_channels=3,
                                                   kernel_size=self.kernel_size, groups=3,
                                                   padding='same', bias=False)

            self.gaussian_filter.weight.data = gaussian_kernel.float()
            self.gaussian_filter.weight.requires_grad = False

    def __call__(self, img, config):
        self.probas = torch.tensor([1 - self.p, self.p], device=self.device, dtype=torch.float).multinomial(
            num_samples=self.batch_size, replacement=True).type(torch.bool)

        with torch.no_grad():
            print(img.dtype)
            print(self.gaussian_filter.weight.dtype)
            img[self.probas] = self.gaussian_filter(img[self.probas])

        torch.cuda.empty_cache()

        return img