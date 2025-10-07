# This code is adapted from the SSRnet-pytorch repository
# Original source: https://github.com/renweidian/SSRnet-pytorch/blob/master/SSR/model3.py
# Accessed on: 2025-04-23

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:32:24 2020

@author: Dian
"""
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy import misc
import cv2
from scipy.io import loadmat
import sys
from matplotlib.pyplot import *
from numpy import *
from torch.nn import functional as F
from torch import nn
import torch
import os
import sys
import torch.utils.data as data
# import metrics
import hdf5storage
import h5py

def reconstruction(net2,R,R_inv,MSI,training_size,stride):
        index_matrix=torch.zeros((R.shape[1],MSI.shape[2],MSI.shape[3])).cuda()
        abundance_t=torch.zeros((R.shape[1],MSI.shape[2],MSI.shape[3])).cuda()
        a=[]
        for j in range(0, MSI.shape[2]-training_size+1, stride):
            a.append(j)
        a.append(MSI.shape[2]-training_size)
        b=[]
        for j in range(0, MSI.shape[3]-training_size+1, stride):
            b.append(j)
        b.append(MSI.shape[3]-training_size)
        for j in a:
            for k in b:
                temp_hrms = MSI[:,:,j:j+training_size, k:k+training_size]
#                temp_hrms=torch.unsqueeze(temp_hrms, 0)
#                print(temp_hrms.shape)
                with torch.no_grad():
                  HSI = net2(R,temp_hrms)
                  HSI=HSI.squeeze()
#                  print(HSI.shape)
                  HSI=torch.clamp(HSI,0,1)
                  abundance_t[:,j:j+training_size, k:k+training_size]= abundance_t[:,j:j+training_size, k:k+training_size]+ HSI
                  index_matrix[:,j:j+training_size, k:k+training_size]= 1+index_matrix[:,j:j+training_size, k:k+training_size]
                
        HSI_recon=abundance_t/index_matrix
        return HSI_recon     
def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F
class HSI_MSI_Data(data.Dataset):
    def __init__(self,train_hrhs_all,train_hrms_all):
        self.train_hrhs_all  = train_hrhs_all
        self.train_hrms_all  = train_hrms_all
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms
    def __len__(self):
        return self.train_hrhs_all.shape[0]
class HSI_MSI_Data1(data.Dataset):
    def __init__(self,path,R,training_size,stride,num):
         imglist=os.listdir(path)
         train_hrhs=[]
         train_hrms=[]
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["b"]
            #img1=img1/img1.max()
            HRHSI=np.transpose(img1,(2,0,1))
            MSI=np.tensordot(R,  HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1]-training_size+1, stride):
                for k in range(0, HRHSI.shape[2]-training_size+1, stride):
                    temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
                    temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
         train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=torch.Tensor(train_hrms)
         print(train_hrhs.shape, train_hrms.shape)
         self.train_hrhs_all  = train_hrhs
         self.train_hrms_all  = train_hrms
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms

    def __len__(self):
        return self.train_hrhs_all.shape[0]
class HSI_MSI_Data2(data.Dataset): 
    def __init__(self,path,R,training_size,stride,num):      
         imglist=os.listdir(path)
         train_hrhs=[]
         # train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=[]
         # train_hrms=torch.Tensor(train_hrms)
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["ref"]
            img1=img1/img1.max()
#            HRHSI=np.transpose(img1,(2,0,1))
#            MSI=np.tensordot(R, HRHSI, axes=([1], [0]))
            HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
            MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
            HRHSI = HRHSI.numpy()
            MSI = MSI.numpy()
            for j in range(0, HRHSI.shape[1]-training_size+1, stride):
                for k in range(0, HRHSI.shape[2]-training_size+1, stride):
                    temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
                    temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
         train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=torch.Tensor(train_hrms)
#         print(train_hrhs.shape, train_hrms.shape)
         self.train_hrhs_all  = torch.Tensor(train_hrhs)
         self.train_hrms_all  = torch.Tensor(train_hrms)
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms

    def __len__(self):
        return self.train_hrhs_all.shape[0]

def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def warm_lr_scheduler(optimizer, init_lr1,init_lr2, warm_iter,iteraion, lr_decay_iter, max_iter, power):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr=init_lr1+iteraion/warm_iter*(init_lr2-init_lr1)
    else:
      lr = init_lr2*(1 - (iteraion-warm_iter)/(max_iter-warm_iter))**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out



def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F

class LossTrainCSS(nn.Module):
    def __init__(self):
        super(LossTrainCSS, self).__init__()
       

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / (label+1e-10)
        mrae = torch.mean(error)
        return mrae


     
class AWCA(nn.Module):
    def __init__(self, channel=31):
        super(AWCA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(channel, 1, bias=False),
#            nn.PReLU(),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Linear(1, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h*w).unsqueeze(1)

        mask = self.conv(x).view(b, 1, h*w)
        mask = self.softmax(mask).unsqueeze(-1)
       
        y = torch.matmul(input_x, mask).view(b, c)
       
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class cycle_fusion(nn.Module):
    def __init__(self,in_chanels,out_chanels):
        super(cycle_fusion, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(in_chanels,out_chanels, 3, 1, 1,bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            ) 
        self.conv2 = nn.Sequential(        
            nn.Conv2d(in_chanels,out_chanels, 3, 1, 1,bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            AWCA(out_chanels),
            ) 
        self.conv3 = nn.Sequential(        
            nn.Conv2d(out_chanels*2,31, 3, 1, 1,bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            AWCA(31),
            ) 
        self.conv4 = nn.Sequential(        
            nn.Conv2d(out_chanels+31,31, 3, 1, 1,bias=True),
            ) 
    def spectral_projection(self,R,MSI, mu,Y1,Y2):
        #print(R.shape,MSI.shape,Y1.shape,1)
        RTR=torch.mm(torch.transpose(R, 1, 0),R)
        #print(RTR.shape,2)
        x = torch.tensordot(MSI, R, dims=([1], [0]))
        #print(x.shape,3)
        x=torch.Tensor.permute(x,(0,3,1,2))+mu*(Y1+Y2)
        #print(x.shape, 4)
        d=torch.inverse(RTR+2*mu*torch.eye(R.shape[1]).cuda())
        #print(d.shape,5)
        x=torch.tensordot(x, torch.inverse(RTR+2*mu*torch.eye(R.shape[1]).cuda()), dims=([1], [1]))
        #print(x.shape,6)
        x=torch.Tensor.permute(x,(0,3,1,2))
        #print(x.shape,7)
        return x  

    def forward(self,R, x1,MSI,mu):
        x2=self.conv2(x1)
        x1= self.conv1(x1)
        x2=torch.cat((x2,x1),1)
        x2=self.conv3(x2)
        x1=torch.cat((x2,x1),1)
        x1=self.conv4(x1)
        x = self.spectral_projection(R,MSI, mu,x1,x2)
        return x    

class CNN_BP_SE5(nn.Module):
    def __init__(self,):
        super(CNN_BP_SE5, self).__init__()
        self.mu=torch.nn.Parameter(torch.ones(7)*1e-5)
        num=32
        self.pro=cycle_fusion(3,64)
        self.pro1=cycle_fusion(31,num)
        self.pro2=cycle_fusion(31,num)
        self.pro3=cycle_fusion(31,num)
        self.pro4=cycle_fusion(31,num)
        self.pro5=cycle_fusion(31,num)
        self.pro6 = cycle_fusion(31, num)

    def forward(self, R,MSI):
        mu=self.mu

        x=MSI
        x=self.pro(R,x, MSI,mu[0])
        x=self.pro1(R,x, MSI,mu[1])
        x=self.pro2(R,x, MSI,mu[2])
        x=self.pro3(R,x, MSI,mu[3])
        x=self.pro4(R,x, MSI,mu[4])
        x=self.pro5(R,x, MSI,mu[5])
        x = self.pro6(R, x, MSI, mu[6])
        return x

# Note: this function is necessary to load the SRF to adapt to the analysis pipeline in our paper.
def load_srf(file_path, dtype=np.float32):
    """
    Load the sensor spectral response function as a numpy array with shape [C, 3].

    Args:
        file_path: Path to the sensor data in .npz format.
        dtype: Data type. Default to np.float32.

    Returns:
        srf: Spectral response function as a numpy array of shape[C, 3].
    """
    data = np.load(file_path)
    srf = np.vstack((data['R'], data['G'], data['B'])).T
    return srf.astype(dtype)

# o=create_F()
# o=torch.Tensor(o).cuda()
# a=torch.rand(1,3,64,64).cuda()
# cnn=CNN_BP_SE5().cuda()
# b=cnn(o,a)
# print(a.shape)