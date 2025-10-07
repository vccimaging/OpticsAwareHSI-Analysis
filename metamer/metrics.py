"""
This code is heavily borrowed MST++ with necessary correction and enrichment.

The original code can be found at
https://github.com/caiyuanhao1998/MST-plus-plus/blob/master/test_develop_code/utils.py
Accessed on: 2025-04-23
"""

import numpy as np
import torch
import torch.nn as nn

EPSILON = 1e-6

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Loss_MRAE(nn.Module):
    """
    Mean Relative Absolute error (MRAE).
    """
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / (label + EPSILON) # avoid 0/0
        mrae = torch.mean(error.reshape(-1)) 
        return mrae

class Loss_RMSE(nn.Module):
    """
    Root Mean Square Error (RMSE).
    """
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1))) 
        return rmse

class Loss_PSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.
    """
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range)

        mse = torch.sum((Ifake-Itrue)**2, dim=(2,3)) / (H*W)
        psnr = 10. * torch.log10((data_range ** 2) / mse)
        return torch.mean(psnr)

class Loss_SAM(nn.Module):
    """
    Spectral Angle Mapper (SAM).
    """
    def __init__(self):
        super(Loss_SAM, self).__init__()

    def forward(self, im_true, im_fake):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.)
        Ifake = im_fake.clamp(0., 1.)

        dot_product = (Ifake * Itrue).sum(dim=1)
        Ifake_norm = Ifake.norm(dim=1)
        Itrue_norm = Itrue.norm(dim=1)
        
        sam = torch.clamp(dot_product / (Ifake_norm * Itrue_norm + EPSILON), -1, 1).acos()
        sam = torch.mean(sam)
        
        return torch.mean(sam)