#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:55:50 2017

@author: max
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import cv2 as cv2



# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

#####################################################
#
# Simple & fast...
#


def rgb2gray(img):
    return np.mean(img, axis=2).astype(np.uint8)

def crop(img, h1, h2, w1, w2):
    return img[h1:h2, w1:w2]

def downsample(img):
    return cv2.resize(img,(84,84))

def breakout_preprocess(img):
    return rgb2gray(downsample(crop(img, 32, 195, 8, 152)))

def rgb2pytorch(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).type(Tensor)

def gray2pytorch(img):
    return torch.from_numpy(img[:,:,None].transpose(2, 0, 1)).unsqueeze(0)
                         



#####################################################
#
# Eher aufwendige Funktionen, die wirkliche Helligkeitswerte... berechnen
# => kann gelöscht werden, wenn nicht mehr gebraucht
    

# Resize an image to output_size
output_size = (84,84)
resize = T.Compose([T.ToPILImage(),
                    T.Resize(output_size, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(env):
    """
    get current screen of breakout and do simple preprocessing
    """
    # TODO: Parametrize strip coordinates
    
    # transpose into torch order (CHW)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    
    # Strip screen at the edges => checked via RGB values of the wall
    screen = screen[:, 32:195, 8:152]
    
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return screen.unsqueeze(0).type(Tensor)

def get_screen_resize(env):
    """
    get current screen of breakout and do simple preprocessing
    """
    # TODO: Parametrize strip coordinates
    
    # transpose into torch order (CHW)
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    
    # Strip screen at the edges => checked via RGB values of the wall
    screen = screen[:, 32:195, 8:152]
    
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(Tensor)

def rgb2gr(img):
    """
    convert rgb to grayscale by averaging channel intensities
    """

    n, ch, w, h = img.size()
    
    r = img[:,0,:,:].unsqueeze(1)
    g = img[:,1,:,:].unsqueeze(1)
    b = img[:,2,:,:].unsqueeze(1)
    
    z = torch.Tensor(n,1,w,h).zero_()
    
    z = 0.21*r + 0.72*g + 0.07*b
    return z

def rgb2y(img):
    """
    convert rgb to luminance (wikipedia)
    """

    n, ch, w, h = img.size()
    
    r = img[:,0,:,:].unsqueeze(1)
    g = img[:,1,:,:].unsqueeze(1)
    b = img[:,2,:,:].unsqueeze(1)
    
    z = torch.Tensor(n,1,w,h).zero_()
    
    z = 0.299*r + 0.587*g + 0.114*b
    return z

def rgb2bw(img):
    """
    convert rgb to black white by checking zero values
    """
    
    n, ch, w, h = img.size()
        
    r = img[:,0,:,:].unsqueeze(1)
    g = img[:,1,:,:].unsqueeze(1)
    b = img[:,2,:,:].unsqueeze(1)
    
    z = torch.Tensor(n,1,w,h).zero_()
    
    z = (r+g+b)!=0
    return z