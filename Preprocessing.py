#!/usr/bin/env python
# coding: utf-8

# In[1]:


import PIL
import time
import copy
import torch
import random
import glob as gb
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm,trange
from time import sleep
from torchsummary import summary
# from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


# In[2]:


"""
Created on Tueday 21:14 13/10/2020

@author: Keira - github.com/Keira. Bai

For pre-process on frame level 
"""


# In[3]:


def LoaData(datapaths, vid_dir,imgpath):
    dataset = []
    expset = []
    exp = [0,0,0,0]
    
    for i in range(len(vid_dir)):
        DataStream = gb.glob(datapaths+vid_dir[i]+imgpath)

        dataset += DataStream
        exp[i] += len(DataStream)
        expset.extend(i for j in range(len(DataStream)))
    return dataset, expset,exp


# In[4]:


def rotation(tran_r,tran_t,img):
    img3 = tran_r(img)
    img3 = tran_t(img3)
    return img3


# In[5]:


def Augmentation(dataset, expset, tran_r,tran_t):
    imgset = []#image tensor
    label = []#exp after augmentation
    expA = [0,0,0,0] 
    
    for img_name,exp in zip(dataset, expset):
        #orginal images
        img = mpimg.imread(img_name)  
        img = Image.fromarray(img) 
        img1 = tran_t(img) 
        imgset.append(img1)
        label.append(exp)
        expA[exp] += 1

        #flip for non-micro 2times
        imgf = transforms.functional.hflip(img)
        img2 = tran_t(imgf)
        imgset.append(img2)
        label.append(exp)
        expA[exp] += 1

        #if the expression is not non-micro-expression
        if exp < 3:
            for i in range(3):# for negative 5times
                imgset.append(rotation(tran_r,tran_t,img))
                label.append(exp)  
                expA[exp] += 1
            if exp == 1:#for positive 6times
                imgset.append(rotation(tran_r,tran_t,img))
                label.append(exp) 
                expA[exp] += 1
            if exp == 2:#for surprise 8times
                for i in range(2):
                    imgset.append(rotation(tran_r,tran_t,img))
                    label.append(exp)  
                    expA[exp] += 1
    return imgset, label, expA


# In[6]:


def reload(imgset, label):
    samp_set = []
    for i,l in zip(imgset, label):
        sample = [i,l]
        samp_set.append(sample)
    return samp_set


# In[7]:


def process_dataloder():
    datapaths = "../SMIC/SMIC_all_cropped/HS/*/"
    vid_dir = ["*/negative/*", "*/positive/*", "*/surprise/*","non_micro/*"]
    imgpath = "/*.bmp"

    use_cuda = torch.cuda.is_available()  
    tran_t = transforms.Compose(
        [transforms.Resize([224,224]), 
         transforms.ToTensor(),     
        ])
    tran_r = transforms.Compose(
        [transforms.RandomRotation(10),        
        ])
    params = {'batch_size': 8, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    dataset, expset, exp = LoaData(datapaths, vid_dir,imgpath) 
    imgset, label, expA = Augmentation(dataset, expset, tran_r,tran_t)
    print("Total samples number:", len(label))
    print("Number of 4 categories:", exp)
    print("Number of 4 categories after augmentation", expA)

    train_label, test_label = train_test_split(label,test_size=1/4, random_state=42)
    train_set = reload(imgset[:len(train_label)], label[:len(train_label)])
    test_set = reload(imgset[len(train_label):], label[len(train_label):])
    train_loader = data.DataLoader(train_set, **params)
    test_loader = data.DataLoader(test_set, **params)

    
    return train_loader, test_loader

